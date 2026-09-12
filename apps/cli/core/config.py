"""The settings the console reads from bijou.toml and BIJOU_* env vars.

The console links nothing in the repo, so it models only the keys it reads: its own [console]
table, and the paths, checkpoint and skills the status bar reports. Every other table in the
shared file belongs to another app and is ignored here.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class ConfigError(Exception):
    """A console setting is out of range."""


class _Foreign(BaseModel):
    """A table owned by another app. Only the keys the console reads are declared."""

    model_config = ConfigDict(extra="ignore")


class Paths(_Foreign):
    base_checkpoints: Path = Path("checkpoints/base")
    adapters: Path = Path("checkpoints/adapters")
    runs: Path = Path("runs")


class Backend(_Foreign):
    checkpoint: str = "nanodiff-150m-sft-alpaca"


class Eval(_Foreign):
    skills: tuple[str, ...] = ("json_extract",)


class Console(BaseModel):
    """The developer console. Every line a unit prints is also appended under log_dir."""

    model_config = ConfigDict(extra="forbid")

    log_lines: int = 5000
    log_dir: Path = Path(".bijou/logs")
    status_interval_secs: float = 5.0
    metrics_interval_secs: float = 2.0

    @model_validator(mode="after")
    def _check(self) -> Console:
        if self.log_lines <= 0:
            raise ConfigError("console.log_lines must be positive")
        if self.status_interval_secs <= 0:
            raise ConfigError("console.status_interval_secs must be positive")
        if self.metrics_interval_secs <= 0:
            raise ConfigError("console.metrics_interval_secs must be positive")
        return self


def _toml_files() -> tuple[Path, ...]:
    override = os.environ.get("BIJOU_CONFIG_FILE")
    return (Path(override),) if override else (Path("bijou.toml"),)


class Config(BaseSettings):
    """What the console reads."""

    model_config = SettingsConfigDict(
        env_prefix="BIJOU_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    console: Console = Field(default_factory=Console)
    paths: Paths = Field(default_factory=Paths)
    backend: Backend = Field(default_factory=Backend)
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

    def adapter_path(self, skill: str) -> Path:
        """Where bijou writes the adapter for one skill."""
        return self.paths.adapters / f"{skill}.pt"

    def full_finetune_path(self, skill: str) -> Path:
        """Where bijou writes the full fine-tune on one skill."""
        base = self.backend.checkpoint or "random"
        return self.paths.base_checkpoints / f"{base}-{skill}-full.pt"
