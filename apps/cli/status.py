"""What the status bar shows: GPUs, the checkpoint, trained artifacts, services, the last run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath

from cli.core.config import Config


@dataclass(frozen=True)
class Gpu:
    """One GPU's load and memory, and the names of the processes holding it."""

    name: str
    used_mib: int
    total_mib: int
    holders: tuple[str, ...] = ()
    util: int = 0
    temp_c: int = 0


@dataclass(frozen=True)
class Machine:
    """What the metrics pane shows about the box itself."""

    load: float
    cpus: int
    mem_used_gib: float
    mem_total_gib: float


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
    # Each compose service that exists, by name: running, starting, unhealthy or restarting.
    services: Mapping[str, str] = field(default_factory=dict)
    machine: Machine | None = None


def parse_gpus(csv: str) -> list[Gpu]:
    """Rows of nvidia-smi name, utilization.gpu, memory.used, memory.total, temperature.gpu."""
    gpus = []
    for row in csv.strip().splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) == 5 and all(part.isdigit() for part in parts[1:]):
            used, total = int(parts[2]), int(parts[3])
            gpus.append(Gpu(parts[0], used, total, util=int(parts[1]), temp_c=int(parts[4])))
    return gpus


def machine() -> Machine | None:
    """Load and memory from the kernel, or None where they cannot be read."""
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            fields[key] = float(rest.split()[0]) / (1024 * 1024)
        total, free = fields["MemTotal"], fields["MemAvailable"]
        return Machine(os.getloadavg()[0], os.cpu_count() or 1, max(total - free, 0.0), total)
    except (OSError, ValueError, KeyError, IndexError):
        return None


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
        query = "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu"
        found = parse_gpus(_nvidia_smi(query))
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


def parse_compose_ps(out: str) -> dict[str, str]:
    """docker compose ps --format json, as service -> running, starting, unhealthy or restarting.

    A container that is up but whose healthcheck has not passed yet is starting, not running: the
    agent cannot reach a model that is still loading. Compose writes one object per line, or one
    array, depending on its version.
    """
    try:
        rows = (
            json.loads(out)
            if out.lstrip().startswith("[")
            else [json.loads(line) for line in out.splitlines() if line.strip()]
        )
    except json.JSONDecodeError:
        return {}
    states = {}
    for row in rows:
        name, state, health = row.get("Service"), row.get("State"), row.get("Health") or ""
        if not name:
            continue
        if state == "running":
            states[name] = "running" if health in ("", "healthy") else health
        elif state in ("restarting", "removing", "paused"):
            states[name] = "restarting"
    return states


def compose_services(compose: Path = Path("deploy/compose.yml")) -> dict[str, str]:
    """What each compose service is doing. Empty when docker is absent or fails."""
    if shutil.which("docker") is None or not compose.exists():
        return {}
    try:
        out = subprocess.run(
            ["docker", "compose", "-f", str(compose), "--profile", "*", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    return parse_compose_ps(out)


def snapshot(cfg: Config) -> Snapshot:
    """The status bar's view of the repo, read from disk, nvidia-smi and docker compose."""
    skills = cfg.eval.skills
    checkpoint = cfg.backend.checkpoint
    present = bool(checkpoint) and (cfg.paths.base_checkpoints / f"{checkpoint}.pt").exists()
    return Snapshot(
        gpus=gpus(),
        checkpoint=checkpoint,
        checkpoint_present=present,
        adapters=sum(cfg.adapter_path(s).exists() for s in skills),
        full_finetunes=sum(cfg.full_finetune_path(s).exists() for s in skills),
        skills=len(skills),
        last_run=last_run(cfg.paths.runs),
        git=git_sha(),
        services=compose_services(),
        machine=machine(),
    )
