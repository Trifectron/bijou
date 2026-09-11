"""Settings from the [evals] table of bijou.toml and BIJOU_EVALS__* env vars."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class Evals(BaseModel):
    """Where the engine is, where cases live, and where reports and the baseline go."""

    model_config = ConfigDict(extra="forbid")

    engine_url: str = "http://127.0.0.1:8200"
    timeout_secs: float = 900.0
    cases_dir: Path = Path("apps/evals/cases")
    baseline_path: Path = Path("apps/evals/baseline.json")
    report_path: Path = Path(".bijou/evals/report.json")
    tolerance: float = 0.0


def _toml_files() -> tuple[Path, ...]:
    override = os.environ.get("BIJOU_CONFIG_FILE")
    return (Path(override),) if override else (Path("bijou.toml"),)


class Config(BaseSettings):
    """What evals reads from the shared file."""

    model_config = SettingsConfigDict(
        env_prefix="BIJOU_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    evals: Evals = Field(default_factory=Evals)

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


def load() -> Evals:
    return Config().evals
