"""Run records.

For a research repo the run record is the product. Every train and eval run
writes one immutable directory under paths.runs holding the resolved config, the
environment, the inputs by digest, and the scores. A number that cannot be
traced to one of these does not go in a table.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.core.determinism import environment
from engine.core.types.errors import EngineError


class RunError(EngineError):
    """A run record could not be written or was written twice."""


def digest(path: Path) -> str:
    """A file's sha256, truncated. Identifies a checkpoint or dataset."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class RunRecord:
    """One run. kind is train or eval; inputs maps a label to a digest."""

    run_id: str
    kind: str
    config: dict[str, Any]
    inputs: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    started_at: str = ""
    finished_at: str = ""
    env: dict[str, str] = field(default_factory=environment)

    @classmethod
    def start(cls, kind: str, config: dict[str, Any], run_id: str | None = None) -> RunRecord:
        # The suffix keeps two runs started in the same second from colliding.
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return cls(
            run_id=run_id or f"{kind}-{stamp}-{secrets.token_hex(3)}",
            kind=kind,
            config=config,
            started_at=datetime.now(UTC).isoformat(),
        )

    def finish(self, **scores: float) -> None:
        self.scores.update(scores)
        self.finished_at = datetime.now(UTC).isoformat()

    def write(self, runs_dir: Path) -> Path:
        """Write the record. Refuses to overwrite an existing run."""
        out = runs_dir / self.run_id
        if out.exists():
            raise RunError(f"run {self.run_id} already exists at {out}")
        out.mkdir(parents=True)
        (out / "record.json").write_text(json.dumps(asdict(self), indent=2, default=str))
        return out


def load_records(runs_dir: Path) -> list[RunRecord]:
    """Every run record under runs_dir, oldest first by start time.

    Run ids carry a random suffix, so filename order does not track creation
    order for runs started in the same second.
    """
    records = [RunRecord(**json.loads(p.read_text())) for p in runs_dir.glob("*/record.json")]
    return sorted(records, key=lambda r: (r.started_at, r.run_id))
