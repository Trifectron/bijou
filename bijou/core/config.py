"""Settings from bijou.toml and BIJOU_* env vars.

Two layers, lowest first: bijou.toml, which is committed, and
BIJOU_<SECTION>__<KEY> from the environment, which wins. Every tunable value
belongs in bijou.toml at its default. Contradictory combinations are rejected
at load rather than clamped at use.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from bijou.core.types import ConfigError


def _toml_files() -> tuple[Path, ...]:
    override = os.environ.get("BIJOU_CONFIG_FILE")
    if override:
        return (Path(override),)
    return (Path("bijou.toml"),)


class Paths(BaseModel):
    """Where checkpoints, adapters, datasets and run records live."""

    base_checkpoints: Path = Path("checkpoints/base")
    adapters: Path = Path("checkpoints/adapters")
    data: Path = Path("data")
    runs: Path = Path("runs")


class Backend(BaseModel):
    """Which base model the run uses. name selects the bijou.backends module."""

    name: str = "nanodiff"
    checkpoint: str = "nanodiff-150m-alpaca"
    device: str = "cuda"
    dtype: str = "bfloat16"
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
    batch_size: int = 16
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
    use_cache: bool = False


class Eval(BaseModel):
    """Scoring. Every number in the composition matrix comes from here."""

    eval_samples: int = 500
    seed: int = 1
    skills: tuple[str, ...] = ("json_extract",)


class Config(BaseSettings):
    """The whole configuration surface."""

    model_config = SettingsConfigDict(
        env_prefix="BIJOU_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    paths: Paths = Field(default_factory=Paths)
    backend: Backend = Field(default_factory=Backend)
    adapter: Adapter = Field(default_factory=Adapter)
    train: Train = Field(default_factory=Train)
    sampling: Sampling = Field(default_factory=Sampling)
    eval: Eval = Field(default_factory=Eval)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003 - signature fixed by pydantic-settings
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=_toml_files()),
        )

    @model_validator(mode="after")
    def validate_combinations(self) -> Config:
        if self.sampling.block_length > self.sampling.gen_length:
            raise ConfigError("sampling.block_length exceeds sampling.gen_length")
        if self.sampling.gen_length % self.sampling.block_length:
            raise ConfigError("sampling.gen_length must be a multiple of sampling.block_length")
        if self.sampling.steps % (self.sampling.gen_length // self.sampling.block_length):
            raise ConfigError(
                "sampling.steps must divide evenly across blocks, or phase "
                "boundaries cannot align with block boundaries"
            )
        if self.train.seed == self.eval.seed:
            raise ConfigError(
                "train.seed equals eval.seed, so the eval split overlaps training data"
            )
        return self


@lru_cache(maxsize=1)
def load() -> Config:
    """The process-wide configuration. Cached; call load.cache_clear() in tests."""
    return Config()
