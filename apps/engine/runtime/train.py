"""Training one adapter on one skill."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch

from engine.adapters import io as adapter_io
from engine.adapters.lora import add, inject, trainable
from engine.backends.nanodiff import NanoDiffBackend
from engine.core.config import Config
from engine.core.determinism import seed_everything
from engine.core.runs import RunRecord, digest
from engine.core.types.diffusion import AdapterSpec, Sample
from engine.skills import load as load_skill


def spec_for(cfg: Config, name: str) -> AdapterSpec:
    """The adapter spec implied by the configuration."""
    return AdapterSpec(
        name=name,
        rank=cfg.adapter.rank,
        alpha=cfg.adapter.alpha,
        targets=cfg.adapter.targets,
        dropout=cfg.adapter.dropout,
    )


def _batches(
    backend: NanoDiffBackend, samples: list[Sample], size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    for start in range(0, len(samples) - size + 1, size):
        chunk = samples[start : start + size]
        encoded = [backend.encode(s.prompt, s.target) for s in chunk]
        yield (
            torch.stack([p for p, _ in encoded]).to(backend.nano.device),
            torch.stack([r for _, r in encoded]).to(backend.nano.device),
        )


def train_adapter(
    cfg: Config,
    skill_name: str,
    full_finetune: bool = False,
    backend: NanoDiffBackend | None = None,
) -> Path:
    """Train one adapter and write its weights and a run record.

    full_finetune trains the base weights instead, which is the upper bound every
    adapter result is compared against.
    """
    seed_everything(cfg.train.seed)
    record = RunRecord.start("train", cfg.model_dump(mode="json"))
    record.notes = f"skill={skill_name} full_finetune={full_finetune}"

    skill = load_skill(skill_name, cfg.paths.data)
    samples = skill.generate(cfg.train.train_samples, cfg.train.seed, split="train")

    backend = backend or NanoDiffBackend(cfg)
    model = backend.build()
    if backend.checkpoint_path is not None:
        record.inputs["base_checkpoint"] = digest(backend.checkpoint_path)

    if full_finetune:
        for p in model.parameters():
            p.requires_grad_(True)
        record.scores["trainable_params"] = float(sum(p.numel() for p in model.parameters()))
    else:
        state = inject(model, cfg.adapter.targets)
        spec = spec_for(cfg, skill_name)
        add(model, spec)
        state.set(skill_name)
        record.scores["trainable_params"] = float(trainable(model, skill_name))

    lr = cfg.train.full_finetune_lr if full_finetune else cfg.train.lr
    optimizer = model.configure_optimizers(
        cfg.train.weight_decay, lr, (0.9, 0.95), backend.nano.device
    )
    # Linear warmup from lr / warmup_steps to lr, then constant.
    warmup = max(cfg.train.warmup_steps, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup))

    step, total = 0, 0.0
    while step < cfg.train.max_steps:
        for prompts, responses in _batches(backend, samples, cfg.train.batch_size):
            if step >= cfg.train.max_steps:
                break
            loss = backend.loss(prompts, responses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total += loss.item()
            step += 1

    record.finish(final_loss=total / max(step, 1), steps=float(step))
    out = record.write(cfg.paths.runs)

    if full_finetune:
        path = cfg.full_finetune_path(skill_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": backend.nano}, path)
    else:
        path = cfg.adapter_path(skill_name)
        adapter_io.save(model, skill_name, path, spec_for(cfg, skill_name))
    (out / "artifact").write_text(str(path))
    return path
