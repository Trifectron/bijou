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

from dataclasses import replace
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

from engine.core.config import Config
from engine.core.protocols import StepHook
from engine.core.types.diffusion import GenerationRequest
from engine.core.types.errors import BackendError

try:
    from nanodiff.config import Config as NanoConfig
    from nanodiff.diffusion import diffusion_loss, forward_process
    from nanodiff.model import NanoDiff
    from nanodiff.sampler import _transfer_schedule
    from nanodiff.sft import (
        SFT_PROMPT_NO_INPUT,
        encode_sft_example,
        sft_forward_process,
        sft_loss,
    )
except ImportError as exc:  # pragma: no cover
    raise BackendError("nanoDiff is not importable; run: git submodule update --init") from exc

__all__ = ["NanoDiffBackend", "Tokenizer", "tiny_config"]


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
    def device(self) -> str:
        return str(self.nano.device)

    @property
    def eot_id(self) -> int:
        return self.enc.eot_token

    def autocast(self) -> torch.autocast:
        """Mixed precision at the configured dtype. A no-op for float32."""
        device_type = "cuda" if str(self.nano.device).startswith("cuda") else "cpu"
        dtype = getattr(torch, self.nano.dtype)
        return torch.autocast(device_type, dtype=dtype, enabled=dtype != torch.float32)

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

    def optimizer(
        self, weight_decay: float, lr: float, betas: tuple[float, float]
    ) -> torch.optim.Optimizer:
        """AdamW from upstream: decay on matrices only."""
        if self.model is None:
            raise BackendError("build the model before making an optimizer")
        optimizer: torch.optim.Optimizer = self.model.configure_optimizers(
            weight_decay, lr, betas, self.nano.device
        )
        return optimizer

    def save(self, path: Path) -> None:
        """The weights and the architecture, as build reads them."""
        if self.model is None:
            raise BackendError("build the model before saving it")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "config": self.nano}, path)

    def prompt_ids(self, prompt: str) -> list[int]:
        """The prompt in the SFT template, truncated from the left like encode."""
        ids = self.enc.encode(SFT_PROMPT_NO_INPUT.format(instruction=prompt))
        return ids[-self.cfg.train.prompt_len :]

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
        with self.autocast():
            logits = self.model(x_t)
            return sft_loss(logits, responses, mask, t)

    def pretrain_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """The pretraining objective. Used only by the parity fixtures."""
        if self.model is None:
            raise BackendError("build the model before computing a loss")
        x_t, mask, t = forward_process(x0, self.nano.mask_token_id)
        with self.autocast():
            return diffusion_loss(self.model(x_t), x0, mask, t)

    @torch.no_grad()
    def generate(self, req: GenerationRequest, on_step: StepHook | None = None) -> str:
        """The response to one instruction, decoded up to the first end-of-text."""
        prompt = torch.tensor([self.prompt_ids(req.prompt)], device=self.nano.device)
        out = self.denoise(prompt, req, on_step)[0, prompt.shape[1] :].tolist()
        if self.eot_id in out:
            out = out[: out.index(self.eot_id)]
        return self.enc.decode([t for t in out if t < self.eot_id])

    @torch.no_grad()
    def denoise(
        self, prompt_ids: torch.Tensor, req: GenerationRequest, on_step: StepHook | None = None
    ) -> torch.Tensor:
        """Low-confidence remasking with semi-autoregressive blocks, as upstream generate.

        Returns the prompt followed by the generated ids. on_step(step, total_steps)
        runs before every denoising step, which is where an ActivationPolicy changes
        the live adapter set. No prefix cache: the cache is invalid when adapters
        change mid-block.
        """
        if self.model is None:
            raise BackendError("build the model before generating")
        model = self.model
        was_training = model.training
        model.eval()
        try:
            with self.autocast():
                return self._denoise(model, prompt_ids, req, on_step)
        finally:
            model.train(was_training)

    def _denoise(
        self,
        model: NanoDiff,
        prompt_ids: torch.Tensor,
        req: GenerationRequest,
        on_step: StepHook | None,
    ) -> torch.Tensor:
        device = prompt_ids.device
        mask_id = self.nano.mask_token_id
        batch, p_len = prompt_ids.shape

        x = torch.full((batch, p_len + req.gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :p_len] = prompt_ids

        block_length = req.block_length or req.gen_length
        blocks = req.gen_length // block_length
        base_steps, extra = divmod(req.steps, blocks)
        step = 0

        for block in range(blocks):
            s0 = p_len + block * block_length
            s1 = s0 + block_length
            block_steps = base_steps + (1 if block < extra else 0)
            counts = _transfer_schedule(block_length, block_steps, device)
            for i in range(block_steps):
                if on_step is not None:
                    on_step(step, req.steps)
                step += 1
                masked = x[:, s0:s1] == mask_id
                if not masked.any():
                    continue
                logits = model(x)[:, s0:s1, :]
                logits[:, :, mask_id] = float("-inf")
                probs = F.softmax(logits.float(), dim=-1)
                if req.temperature > 0:
                    sampling = F.softmax(logits.float() / req.temperature, dim=-1)
                    prediction = torch.multinomial(sampling.view(-1, sampling.size(-1)), 1)
                    prediction = prediction.view(batch, -1)
                else:
                    prediction = logits.argmax(dim=-1)
                confidence = probs.gather(-1, prediction.unsqueeze(-1)).squeeze(-1)
                confidence = torch.where(
                    masked, confidence, torch.full_like(confidence, float("-inf"))
                )
                k = int(counts[i])
                if k <= 0:
                    continue
                chosen = confidence.topk(k, dim=1).indices
                commit = torch.zeros_like(masked)
                commit.scatter_(1, chosen, True)
                x[:, s0:s1] = torch.where(commit, prediction, x[:, s0:s1])
        return x
