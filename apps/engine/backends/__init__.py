"""Vendored model implementations. The only package importing third_party.

Importing this package puts the vendored nanoDiff on sys.path, so any module
below it can import nanodiff regardless of which one is loaded first. create
builds the backend backend.name selects; each backend module is imported only
when it is selected.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from engine.core.config import Config
from engine.core.protocols import Backend

VENDOR = Path(__file__).resolve().parents[3] / "third_party" / "nanoDiff"

if VENDOR.is_dir() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


def _nanodiff(cfg: Config) -> Backend:
    from engine.backends.nanodiff import NanoDiffBackend

    return NanoDiffBackend(cfg)


FACTORIES: dict[str, Callable[[Config], Backend]] = {"nanodiff": _nanodiff}


def create(cfg: Config) -> Backend:
    """The backend named by backend.name."""
    return FACTORIES[cfg.backend.name](cfg)
