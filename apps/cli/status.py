"""What the status bar shows: GPUs, the checkpoint, trained artifacts, services, the last run."""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path, PurePath

from cli.core.config import Config


@dataclass(frozen=True)
class Gpu:
    """One GPU's memory, and the names of the processes holding it."""

    name: str
    used_mib: int
    total_mib: int
    holders: tuple[str, ...] = ()


@dataclass(frozen=True)
class Snapshot:
    """Everything the status bar shows, gathered at one moment."""

    gpus: tuple[Gpu, ...]
    checkpoint: str
    checkpoint_present: bool
    adapters: int
    full_finetunes: int
    skills: int
    last_run: str
    git: str
    services: tuple[tuple[str, bool], ...] = ()


def parse_gpus(csv: str) -> list[Gpu]:
    """Rows of nvidia-smi name, memory.used, memory.total in csv, noheader, nounits form."""
    gpus = []
    for row in csv.strip().splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            gpus.append(Gpu(parts[0], int(parts[1]), int(parts[2])))
    return gpus


def parse_holders(csv: str) -> tuple[str, ...]:
    """Process names from nvidia-smi pid, process_name rows, one per process."""
    names = []
    for row in csv.strip().splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) >= 2 and parts[1]:
            names.append(PurePath(parts[1]).name)
    return tuple(names)


def _nvidia_smi(*query: str) -> str:
    return subprocess.run(
        ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout


def gpus() -> tuple[Gpu, ...]:
    """Every visible NVIDIA GPU. Empty when nvidia-smi is absent or fails."""
    if shutil.which("nvidia-smi") is None:
        return ()
    try:
        found = parse_gpus(_nvidia_smi("--query-gpu=name,memory.used,memory.total"))
        holders = parse_holders(_nvidia_smi("--query-compute-apps=pid,process_name"))
    except (subprocess.SubprocessError, OSError):
        return ()
    if found:
        found[0] = replace(found[0], holders=holders)
    return tuple(found)


def git_sha() -> str:
    """The working tree's commit, with -dirty appended when it has changes."""

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()

    try:
        sha = git("rev-parse", "--short", "HEAD")
        return f"{sha}-dirty" if git("status", "--porcelain") else sha
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def last_run(runs: Path) -> str:
    """The kind and start time of the newest run record, or empty."""
    newest: tuple[str, str] | None = None
    for path in runs.glob("*/record.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        key = (str(record.get("started_at", "")), str(record.get("kind", "")))
        if newest is None or key > newest:
            newest = key
    if newest is None:
        return ""
    return f"{newest[1]} {newest[0][5:16].replace('T', ' ')}"


def up(url: str, timeout: float = 0.5) -> bool:
    """Whether a service answers its health route."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback URL from config
            return bool(response.status == 200)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def snapshot(cfg: Config) -> Snapshot:
    """The status bar's view of the repo, read from disk, nvidia-smi and the health routes."""
    skills = cfg.eval.skills
    checkpoint = cfg.backend.checkpoint
    present = bool(checkpoint) and (cfg.paths.base_checkpoints / f"{checkpoint}.pt").exists()
    services = (
        ("serve", up(f"http://{cfg.serve.host}:{cfg.serve.port}/health")),
        ("engine", up(f"http://{cfg.agent.http.host}:{cfg.agent.http.port}/health")),
    )
    return Snapshot(
        gpus=gpus(),
        checkpoint=checkpoint,
        checkpoint_present=present,
        adapters=sum(cfg.adapter_path(s).exists() for s in skills),
        full_finetunes=sum(cfg.full_finetune_path(s).exists() for s in skills),
        skills=len(skills),
        last_run=last_run(cfg.paths.runs),
        git=git_sha(),
        services=services,
    )
