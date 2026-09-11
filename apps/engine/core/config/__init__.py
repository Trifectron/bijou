"""Settings from bijou.toml and BIJOU_* env vars.

Two layers, lowest first: bijou.toml, which is committed, and BIJOU_<TABLE>__<KEY> from the
environment, which wins. BIJOU_CONFIG_FILE points elsewhere. Every tunable value belongs in
bijou.toml at its default; secrets and per-machine URLs live in .env. Contradictory combinations
are rejected at load rather than clamped at use.

  model  the diffusion model's tables, unprefixed, plus [bank] and [collect]
  agent  the [agent.*] tables

A table another app owns, such as [evals] or [console], is ignored. Every table declared here
rejects an unknown key.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from engine.core.config.agent import (
    AgentConfig,
    Llm,
    Loop,
    Mcp,
    McpServer,
    Patterns,
    Planning,
    Policy,
    Prompt,
    Sessions,
    Skills,
    Tools,
    Trace,
)
from engine.core.config.diffusion import (
    Adapter,
    Backend,
    Bank,
    Collect,
    Condition,
    Eval,
    Paths,
    Prompting,
    Sampling,
    Train,
)
from engine.core.types.errors import ConfigError

__all__ = [
    "Adapter",
    "AgentConfig",
    "Backend",
    "Bank",
    "Collect",
    "Condition",
    "Config",
    "Eval",
    "Llm",
    "Loop",
    "Mcp",
    "McpServer",
    "Paths",
    "Patterns",
    "Planning",
    "Policy",
    "Prompt",
    "Prompting",
    "Sampling",
    "Sessions",
    "Skills",
    "Telemetry",
    "Tools",
    "Trace",
    "Train",
    "load",
]


class Telemetry(BaseModel):
    """Spans over OTLP, read by Phoenix, and Prometheus metrics. otlp_endpoint is per-machine and
    lives in .env as BIJOU_TELEMETRY__OTLP_ENDPOINT; empty sends no spans. engine chat serves the
    metrics at metrics_host:metrics_port while it runs; port 0 serves none."""

    model_config = ConfigDict(extra="forbid")

    otlp_endpoint: str = ""
    service_name: str = "bijou-engine"
    sample_ratio: float = 1.0
    export_timeout_secs: float = 10.0
    metrics_host: str = "127.0.0.1"
    metrics_port: int = 9464

    @model_validator(mode="after")
    def _check(self) -> Telemetry:
        if not 0 <= self.sample_ratio <= 1:
            raise ConfigError("telemetry.sample_ratio must lie in [0, 1]")
        if self.export_timeout_secs <= 0:
            raise ConfigError("telemetry.export_timeout_secs must be positive")
        if not 0 <= self.metrics_port < 65536:
            raise ConfigError("telemetry.metrics_port must be a TCP port, or 0 for none")
        return self


def _toml_files() -> tuple[Path, ...]:
    override = os.environ.get("BIJOU_CONFIG_FILE")
    return (Path(override),) if override else (Path("bijou.toml"),)


class Config(BaseSettings):
    """The whole engine configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BIJOU_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    paths: Paths = Field(default_factory=Paths)
    backend: Backend = Field(default_factory=Backend)
    adapter: Adapter = Field(default_factory=Adapter)
    train: Train = Field(default_factory=Train)
    sampling: Sampling = Field(default_factory=Sampling)
    eval: Eval = Field(default_factory=Eval)
    prompting: Prompting = Field(default_factory=Prompting)
    bank: Bank = Field(default_factory=Bank)
    collect: Collect = Field(default_factory=Collect)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    telemetry: Telemetry = Field(default_factory=Telemetry)

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
        if self.train.train_samples < self.train.batch_size:
            raise ConfigError("train.train_samples is smaller than one train.batch_size batch")
        if self.train.seed == self.eval.seed:
            raise ConfigError(
                "train.seed equals eval.seed, so the eval split overlaps training data"
            )
        if self.prompting.seed in (self.train.seed, self.eval.seed):
            raise ConfigError("prompting.seed must differ from train.seed and eval.seed")
        if not self.eval.conditions:
            raise ConfigError("eval.conditions is empty, the matrix would score nothing")
        return self

    def adapter_path(self, skill: str) -> Path:
        """Where the adapter trained on one skill is written."""
        return self.paths.adapters / f"{skill}.pt"

    def full_finetune_path(self, skill: str) -> Path:
        """Where the full fine-tune on one skill is written, beside the base checkpoints."""
        base = self.backend.checkpoint or "random"
        return self.paths.base_checkpoints / f"{base}-{skill}-full.pt"

    @property
    def proposals_dir(self) -> Path:
        """Where skill specs wait for review before collection."""
        return self.paths.data / "proposals"


@lru_cache(maxsize=1)
def load() -> Config:
    """The process-wide configuration. Cached; call load.cache_clear() in tests."""
    return Config()
