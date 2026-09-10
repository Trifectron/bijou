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

import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from bijou.core.config import Config
from bijou.core.types import BackendError, GenerationRequest

_VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "nanoDiff"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

try:
    import tiktoken
    from nanodiff.config import Config as NanoConfig
    from nanodiff.diffusion import diffusion_loss, forward_process
    from nanodiff.model import NanoDiff
    from nanodiff.sampler import _transfer_schedule
    from nanodiff.sft import encode_sft_example, sft_forward_process, sft_loss
except ImportError as exc:  # pragma: no cover
    raise BackendError("nanoDiff is not importable; run: git submodule update --init") from exc

__all__ = ["NanoDiffBackend", "tiny_config"]

StepHook = Callable[[int, int], object]


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

    def __init__(self, cfg: Config, nano: NanoConfig | None = None) -> None:
        self.cfg = cfg
        self.nano = nano or self._nano_config()
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot_id = self.enc.eot_token
        self.model: NanoDiff | None = None

    def _nano_config(self) -> NanoConfig:
        return NanoConfig(
            device=self.cfg.backend.device,
            dtype=self.cfg.backend.dtype,
            compile=self.cfg.backend.compile,
        )

    def build(self) -> NanoDiff:
        """Construct the model and load the base checkpoint when one is configured."""
        model = NanoDiff(self.nano).to(self.nano.device)
        path = self.cfg.paths.base_checkpoints / f"{self.cfg.backend.checkpoint}.pt"
        if path.exists():
            blob = torch.load(path, map_location=self.nano.device, weights_only=False)
            model.load_state_dict(blob.get("model", blob))
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
