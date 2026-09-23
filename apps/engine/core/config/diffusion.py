"""The diffusion model's settings: paths, backend, adapters, training, sampling, evaluation,
the tuned-prompt baseline, the skill server and collection."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

from engine.core.types.errors import ConfigError

Condition = Literal["adapters", "tuned-prompt", "full-finetune"]
BackendName = Literal["nanodiff"]


class Paths(BaseModel):
    """Where checkpoints, adapters, datasets and run records live."""

    base_checkpoints: Path = Path("checkpoints/base")
    adapters: Path = Path("checkpoints/adapters")
    data: Path = Path("data")
    runs: Path = Path("runs")


class Backend(BaseModel):
    """Which base model the run uses. name selects the engine.backends module.

    checkpoint names a file in paths.base_checkpoints without its extension. An
    empty checkpoint starts the model from random weights.
    """

    name: BackendName = "nanodiff"
    checkpoint: str = "nanodiff-150m-sft-alpaca"
    device: str = "cuda"
    dtype: Literal["float32", "bfloat16"] = "bfloat16"
    compile: bool = False


class Adapter(BaseModel):
    """Defaults for a newly created adapter."""

    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    targets: tuple[str, ...] = (
        "attn.qkv",
        "attn.proj",
        "mlp.w1",
        "mlp.w2",
        "mlp.w3",
    )

    @model_validator(mode="after")
    def _check(self) -> Adapter:
        if self.rank <= 0:
            raise ConfigError("adapter.rank must be positive")
        if not self.targets:
            raise ConfigError("adapter.targets is empty, nothing would be adapted")
        if any("lm_head" in t for t in self.targets):
            raise ConfigError("adapter.targets includes lm_head, which is weight-tied to tok_emb")
        return self


class Train(BaseModel):
    """The adapter fine-tuning loop."""

    lr: float = 1e-3
    full_finetune_lr: float = 5e-5
    batch_size: int = 8
    max_steps: int = 2000
    warmup_steps: int = 100
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    prompt_len: int = 256
    response_len: int = 256
    train_samples: int = 20_000
    seed: int = 0


class Sampling(BaseModel):
    """The reverse process used for every eval generation."""

    steps: int = 64
    gen_length: int = 128
    block_length: int = 64
    temperature: float = 0.0


class Eval(BaseModel):
    """Scoring. Every number in the composition matrix comes from here."""

    eval_samples: int = 500
    seed: int = 1
    skills: tuple[str, ...] = ("json_extract",)
    conditions: tuple[Condition, ...] = ("adapters", "tuned-prompt", "full-finetune")


class Prompting(BaseModel):
    """The tuned-prompt baseline: candidates scored on a dev split, the best one evaluated."""

    shots: tuple[int, ...] = (0, 3)
    dev_samples: int = 100
    seed: int = 2

    @model_validator(mode="after")
    def _check(self) -> Prompting:
        if not self.shots or any(k < 0 for k in self.shots):
            raise ConfigError("prompting.shots needs at least one count, none negative")
        if self.dev_samples <= 0:
            raise ConfigError("prompting.dev_samples must be positive")
        return self


class Bank(BaseModel):
    """The skill bank: one base model with every trained LoRA attached, in the agent's process."""

    model_config = ConfigDict(extra="forbid")

    max_gen_length: int = 512

    @model_validator(mode="after")
    def _check(self) -> Bank:
        if self.max_gen_length <= 0:
            raise ConfigError("bank.max_gen_length must be positive")
        return self


class Collect(BaseModel):
    """LoRA collection: a teacher LLM writes the examples a dataset skill trains on.

    base_url and api_key are per-machine and live in .env as BIJOU_COLLECT__BASE_URL and
    BIJOU_COLLECT__API_KEY.
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: SecretStr = SecretStr("")
    model: str = "Qwen/Qwen3-4B-GGUF:Q4_K_M"
    thinking: bool = False
    temperature: float = 0.8
    max_tokens: int = 4096
    timeout_secs: float = 180.0
    examples: int = 300
    per_request: int = 10
    max_rounds: int = 100
    dev_ratio: float = 0.1
    eval_ratio: float = 0.2
    seed: int = 3

    @model_validator(mode="after")
    def _check(self) -> Collect:
        if self.examples < 3:
            raise ConfigError("collect.examples must leave at least one example per split")
        if self.per_request <= 0 or self.max_rounds <= 0:
            raise ConfigError("collect.per_request and collect.max_rounds must be positive")
        if not (0 < self.dev_ratio < 1 and 0 < self.eval_ratio < 1):
            raise ConfigError("collect.dev_ratio and collect.eval_ratio must lie in (0, 1)")
        if self.dev_ratio + self.eval_ratio >= 1:
            raise ConfigError("collect.dev_ratio plus collect.eval_ratio leaves no train split")
        return self
