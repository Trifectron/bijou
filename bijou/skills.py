"""Skill manifests: an adapter plus the metadata that makes it auditable.

A bare .safetensors file is not a skill. Without a recorded base checkpoint,
training set, and eval result, an adapter is an unfalsifiable claim -- and the
whole project rests on claims about what adapters do and do not preserve when
composed. The manifest is the unit that makes RQ1/RQ2 measurable.

Deliberately NOT in the manifest: `good_for` / `avoid_for` prose. Natural
language applicability hints are for a router we have not built and have no
evidence we need. Add them when there is a router that consumes them.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    name: str
    base_checkpoint: str        # exact base ckpt -- adapters are not portable across bases
    rank: int
    alpha: float
    targets: tuple[str, ...]
    dataset: str                # path or generator id producing the training set
    eval: str                   # the auto-gradable eval this skill is scored on
    adapter_path: Path | None = None
    scores: dict[str, float] = field(default_factory=dict)  # eval name -> score

    @classmethod
    def load(cls, manifest_path: str | Path) -> "Skill":
        path = Path(manifest_path)
        raw = tomllib.loads(path.read_text())
        skill = raw["skill"]
        adapter = skill.get("adapter_path")
        return cls(
            name=skill["name"],
            base_checkpoint=skill["base_checkpoint"],
            rank=int(skill["rank"]),
            alpha=float(skill.get("alpha", skill["rank"])),
            targets=tuple(skill.get("targets", [])) or None,
            dataset=skill["dataset"],
            eval=skill["eval"],
            adapter_path=(path.parent / adapter) if adapter else None,
            scores=dict(raw.get("scores", {})),
        )


def load_registry(root: str | Path = "skills") -> dict[str, Skill]:
    """Load every skills/*/manifest.toml, keyed by skill name."""
    registry = {}
    for manifest in sorted(Path(root).glob("*/manifest.toml")):
        skill = Skill.load(manifest)
        if skill.name in registry:
            raise ValueError(f"duplicate skill name {skill.name!r}")
        registry[skill.name] = skill
    return registry
