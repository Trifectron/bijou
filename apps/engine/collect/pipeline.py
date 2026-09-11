"""Collection: an approved spec in, a dataset skill on disk and a run record out.

The spec's own pairs are kept, the teacher writes the rest, duplicates by input are dropped, and
what remains is shuffled by collect.seed and split into train, dev and eval files. The split is
fixed here, once, so no later seed can move an example across it.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from engine.collect.spec import Pair, SkillSpec, SpecError
from engine.collect.teacher import Teacher
from engine.core.config import Config
from engine.core.runs import RunRecord
from engine.skills import KNOWN
from engine.skills.dataset import SPLITS, DatasetSkill, dataset_dir


def _key(text: str) -> str:
    return " ".join(text.lower().split())


def gather(cfg: Config, spec: SkillSpec, teacher: Teacher, n: int) -> list[Pair]:
    """Up to n unique examples: the spec's pairs first, then the teacher's."""
    pairs: list[Pair] = []
    seen: set[str] = set()

    def keep(batch: list[Pair]) -> None:
        for pair in batch:
            key = _key(pair.input)
            if key not in seen and len(pairs) < n:
                seen.add(key)
                pairs.append(pair)

    keep(spec.pairs)
    for _ in range(cfg.collect.max_rounds):
        if len(pairs) >= n:
            break
        keep(teacher.write(spec, min(cfg.collect.per_request, n - len(pairs)), pairs))
    return pairs


def split(cfg: Config, pairs: list[Pair]) -> dict[str, list[Pair]]:
    """Shuffle by collect.seed and cut into eval, dev and train, in that order."""
    shuffled = list(pairs)
    random.Random(cfg.collect.seed).shuffle(shuffled)
    n_eval = max(1, round(len(shuffled) * cfg.collect.eval_ratio))
    n_dev = max(1, round(len(shuffled) * cfg.collect.dev_ratio))
    return {
        "eval": shuffled[:n_eval],
        "dev": shuffled[n_eval : n_eval + n_dev],
        "train": shuffled[n_eval + n_dev :],
    }


def collect(
    cfg: Config, spec: SkillSpec, teacher: Teacher, n: int | None = None, overwrite: bool = False
) -> Path:
    """Write the dataset skill for an approved spec and a collect run record. Returns its dir."""
    if not spec.approved:
        raise SpecError(f"spec {spec.name} is not approved; read it, then set approved to true")
    if spec.name in KNOWN:
        raise SpecError(f"{spec.name} is a built-in skill")
    out = dataset_dir(cfg.paths.data, spec.name)
    if out.exists() and not overwrite:
        raise SpecError(f"{out} exists; pass --overwrite to collect it again")

    want = n or cfg.collect.examples
    record = RunRecord.start("collect", cfg.model_dump(mode="json"))
    record.notes = f"skill={spec.name} source={spec.source}"
    pairs = gather(cfg, spec, teacher, want)
    parts = split(cfg, pairs)
    if not parts["train"]:
        raise SpecError(f"only {len(pairs)} examples collected, too few to leave a train split")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    meta = {"name": spec.name, "description": spec.description, "instruction": spec.instruction}
    (out / "skill.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name in SPLITS:
        lines = [json.dumps(p.model_dump()) for p in parts[name]]
        (out / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    DatasetSkill.read(out)

    record.finish(
        requested=float(want),
        collected=float(len(pairs)),
        **{f"{name}_examples": float(len(parts[name])) for name in SPLITS},
    )
    run_dir = record.write(cfg.paths.runs)
    (run_dir / "artifact").write_text(str(out))
    return out
