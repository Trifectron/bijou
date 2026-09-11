"""The nanoDiff backend.

The only module that imports third_party.nanoDiff. Its internals stay here:
Config construction, the SFT encoder, the masking objective, and the reverse
process. Moving to a larger diffusion LM means writing a sibling module, not
editing the rest of the package.

The denoising loop is reimplemented rather than reused because upstream
generate takes no per-step hook, and the ActivationPolicy seam needs one. The
loop must match upstream token for token when no adapter is active; the parity
test asserts it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

from bijou.core.config import Config
from bijou.core.types import BackendError, GenerationRequest

try:
    from nanodiff.config import Config as NanoConfig
    from nanodiff.diffusion import diffusion_loss, forward_process
    from nanodiff.model import NanoDiff
    from nanodiff.sampler import _transfer_schedule
    from nanodiff.sft import encode_sft_example, sft_forward_process, sft_loss
except ImportError as exc:  # pragma: no cover
    raise BackendError("nanoDiff is not importable; run: git submodule update --init") from exc

__all__ = ["NanoDiffBackend", "Tokenizer", "tiny_config"]

StepHook = Callable[[int, int], object]


class Tokenizer(Protocol):
    """What the backend needs of a tokenizer. tiktoken's gpt2 encoding satisfies it."""

    eot_token: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


def tiny_config(**overrides: object) -> NanoConfig:
    """A 2-layer model that fits on CPU. The fixture every test builds on."""
    base = {
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "block_size": 128,
        "device": "cpu",
        "dtype": "float32",
        "compile": False,
    }
    return NanoConfig(**(base | overrides))


class NanoDiffBackend:
    """Builds, trains and samples a nanoDiff model. Satisfies Backend."""

    def __init__(
        self, cfg: Config, nano: NanoConfig | None = None, tokenizer: Tokenizer | None = None
    ) -> None:
        self.cfg = cfg
        self.nano = nano or self._nano_config()
        self.model: NanoDiff | None = None
        self._enc = tokenizer

    @property
    def enc(self) -> Tokenizer:
        """The tokenizer, built on first use.

        tiktoken fetches its vocabulary over the network the first time, so
        construction is deferred and a tokenizer can be injected instead.
        """
        if self._enc is None:
            import tiktoken

            self._enc = tiktoken.get_encoding("gpt2")
        return self._enc

    @property
    def eot_id(self) -> int:
        return self.enc.eot_token

    def _nano_config(self) -> NanoConfig:
        return NanoConfig(
            device=self.cfg.backend.device,
            dtype=self.cfg.backend.dtype,
            compile=self.cfg.backend.compile,
        )

    @property
    def checkpoint_path(self) -> Path | None:
        """The configured base checkpoint, or None when the model starts from random weights."""
        if not self.cfg.backend.checkpoint:
            return None
        return self.cfg.paths.base_checkpoints / f"{self.cfg.backend.checkpoint}.pt"

    def build(self) -> NanoDiff:
        """Construct the model. A base checkpoint sets both its architecture and its weights."""
        path = self.checkpoint_path
        weights = None
        if path is not None:
            if not path.exists():
                raise BackendError(f"base checkpoint {path} is missing; run: just checkpoints")
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(blob, dict) or "config" not in blob or "model" not in blob:
                raise BackendError(f"{path} is not a nanoDiff checkpoint with model and config")
            self.nano = replace(
                blob["config"],
                device=self.cfg.backend.device,
                dtype=self.cfg.backend.dtype,
                compile=self.cfg.backend.compile,
            )
            # Checkpoints saved from a compiled model prefix every key with _orig_mod.
            weights = {k.removeprefix("_orig_mod."): v for k, v in blob["model"].items()}
        model = NanoDiff(self.nano)
        if weights is not None:
            model.load_state_dict(weights)
        model = model.to(self.nano.device)
        self.model = model
        return model

    def encode(self, prompt: str, target: str) -> tuple[torch.Tensor, torch.Tensor]:
        """One sample as fixed-width prompt and response ids."""
        prompt_ids, response_ids = encode_sft_example(
            instruction=prompt,
            input_text="",
            output=target,
            enc=self.enc,
            prompt_len=self.cfg.train.prompt_len,
            response_len=self.cfg.train.response_len,
            eot_id=self.eot_id,
        )
        return torch.tensor(prompt_ids), torch.tensor(response_ids)

    def loss(self, prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """The SFT objective: response-only masking, 1/t weighted cross entropy."""
        if self.model is None:
            raise BackendError("build the model before computing a loss")
        x_t, mask, t = sft_forward_process(prompts, responses, self.nano.mask_token_id)
        logits = self.model(x_t)
        return sft_loss(logits, responses, mask, t)

    def pretrain_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """The pretraining objective. Used only by the parity fixtures."""
        if self.model is None:
            raise BackendError("build the model before computing a loss")
        x_t, mask, t = forward_process(x0, self.nano.mask_token_id)
        return diffusion_loss(self.model(x_t), x0, mask, t)

    @torch.no_grad()
    def generate(self, req: GenerationRequest, on_step: StepHook | None = None) -> str:
        """Low-confidence remasking with semi-autoregressive blocks.

        on_step(step, total_steps) runs before every forward, which is where an
        ActivationPolicy changes the live adapter set. No prefix cache: the cache
        is invalid when adapters change mid-block.
        """
        if self.model is None:
            raise BackendError("build the model before generating")
        device = self.nano.device
        mask_id = self.nano.mask_token_id

        prompt = self.enc.encode(req.prompt)
        prompt = prompt[-self.cfg.train.prompt_len :]
        prompt_ids = torch.tensor([prompt], device=device)
        p_len = prompt_ids.shape[1]

        x = torch.full((1, p_len + req.gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :p_len] = prompt_ids

        block_length = req.block_length or req.gen_length
        blocks = req.gen_length // block_length
        steps_per_block = req.steps // blocks
        step = 0

        for block in range(blocks):
            s0 = p_len + block * block_length
            s1 = s0 + block_length
            counts = _transfer_schedule(block_length, steps_per_block, device)
            for i in range(steps_per_block):
                if on_step is not None:
                    on_step(step, req.steps)
                step += 1
                logits = self.model(x)
                probs = F.softmax(logits.float(), dim=-1)
                confidence, prediction = probs.max(dim=-1)
                if req.temperature > 0:
                    prediction = torch.multinomial(
                        F.softmax(logits.float() / req.temperature, dim=-1).squeeze(0), 1
                    ).view(1, -1)
                masked = x[:, s0:s1] == mask_id
                scored = torch.where(
                    masked, confidence[:, s0:s1], torch.full_like(confidence[:, s0:s1], -1.0)
                )
                k = int(min(counts[i].item(), int(masked.sum().item())))
                if k <= 0:
                    continue
                chosen = scored.topk(k, dim=-1).indices
                block_view = x[:, s0:s1]
                block_view.scatter_(1, chosen, prediction[:, s0:s1].gather(1, chosen))

        out = x[0, p_len:].tolist()
        if self.eot_id in out:
            out = out[: out.index(self.eot_id)]
        return self.enc.decode(out)
