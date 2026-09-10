"""Seeding and environment capture.

A number in the composition matrix is worth nothing if the run that produced it
cannot be repeated. Seeding happens in one place so no experiment can forget it.
"""

from __future__ import annotations

import os
import random
import subprocess


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch, and put cuBLAS in a reproducible mode."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def git_sha() -> str:
    """The working tree's commit, with -dirty appended when it has changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return f"{sha}-dirty" if dirty.strip() else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def environment() -> dict[str, str]:
    """Versions that change results. Recorded in every run manifest."""
    env = {"git_sha": git_sha()}
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda or "cpu"
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch"] = "absent"
    return env
