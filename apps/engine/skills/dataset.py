"""Dataset skills: a skill whose data was collected onto disk rather than generated.

bijou collect writes one directory per skill under paths.data/skills:

  skill.json     name, description, instruction
  train.jsonl    one {"input": ..., "output": ...} per line
  dev.jsonl      the split the tuned-prompt baseline is chosen on
  eval.jsonl     the held-out split every reported number comes from

The splits are separate files, so no seed can make them overlap. generate(n, seed, split) reads
the named split and draws up to n examples from it in an order fixed by seed.

The grader is deterministic. An output that parses as the same JSON value as the target passes;
otherwise outputs are compared after collapsing whitespace and case, and the value is token F1.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from engine.core.types.diffusion import Sample, Score, Split
from engine.core.types.errors import EngineError

SPLITS: tuple[Split, ...] = ("train", "dev", "eval")
PROMPT = "{instruction}\n\n{text}"
_NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class DatasetError(EngineError):
    """A dataset skill directory is missing a file or holds a malformed line."""


def dataset_dir(data: Path, name: str) -> Path:
    """Where the dataset skill of that name lives."""
    return data / "skills" / name


def dataset_names(data: Path) -> tuple[str, ...]:
    """Every dataset skill under data, sorted."""
    root = data / "skills"
    if not root.is_dir():
        return ()
    return tuple(sorted(p.parent.name for p in root.glob("*/skill.json")))


def valid_name(name: str) -> bool:
    """A skill name is lower snake case, which is what adapter files and URLs carry."""
    return bool(_NAME.match(name))


def _read_split(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise DatasetError(f"{path} is missing")
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rows.append((str(row["input"]), str(row["output"])))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DatasetError(f"{path}:{number} is not an input and output object") from exc
    return rows


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _f1(got: str, want: str) -> float:
    a, b = _tokens(got), _tokens(want)
    if not a or not b:
        return float(a == b)
    common = sum(min(a.count(t), b.count(t)) for t in set(a))
    if not common:
        return 0.0
    precision, recall = common / len(a), common / len(b)
    return 2 * precision * recall / (precision + recall)


def _as_json(text: str) -> object | None:
    try:
        value: object = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value


@dataclass(frozen=True)
class DatasetSkill:
    """A collected skill. Satisfies core.protocols.Skill."""

    NAME: str
    DESCRIPTION: str
    INSTRUCTIONS: tuple[str, ...]
    PROMPT: str = PROMPT
    splits: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    @classmethod
    def read(cls, directory: Path) -> DatasetSkill:
        """Read skill.json and every split from one directory."""
        meta_path = directory / "skill.json"
        if not meta_path.exists():
            raise DatasetError(f"{meta_path} is missing")
        meta = json.loads(meta_path.read_text())
        name = str(meta.get("name", ""))
        if name != directory.name:
            raise DatasetError(f"{meta_path} names {name!r} but lives in {directory.name!r}")
        return cls(
            NAME=name,
            DESCRIPTION=str(meta.get("description", "")),
            INSTRUCTIONS=(str(meta["instruction"]),),
            splits={s: _read_split(directory / f"{s}.jsonl") for s in SPLITS},
        )

    def generate(self, n: int, seed: int, split: Split = "train") -> list[Sample]:
        """Up to n examples of one split, in an order fixed by seed."""
        rows = list(self.splits.get(split, []))
        if not rows:
            raise DatasetError(f"skill {self.NAME} has no {split} examples")
        random.Random(seed).shuffle(rows)
        return [
            Sample(
                id=f"{self.NAME}-{split}-{seed}-{i}",
                prompt=self.PROMPT.format(instruction=self.INSTRUCTIONS[0], text=text),
                target=target,
                meta={"text": text},
            )
            for i, (text, target) in enumerate(rows[:n])
        ]

    def grade(self, sample: Sample, output: str) -> Score:
        """Same JSON value, or the same text up to whitespace and case."""
        want = _as_json(sample.target)
        if want is not None and not isinstance(want, str):
            got = _as_json(output)
            if got == want:
                return Score(passed=True, value=1.0)
            if isinstance(want, dict) and isinstance(got, dict):
                right = sum(1 for k, v in want.items() if got.get(k) == v)
                return Score(
                    passed=False,
                    value=right / max(len(want), 1),
                    detail=f"{right}/{len(want)} keys",
                )
            return Score(passed=False, value=0.0, detail="not the same JSON value")
        if " ".join(_tokens(output)) == " ".join(_tokens(sample.target)):
            return Score(passed=True, value=1.0)
        value = _f1(output, sample.target)
        return Score(passed=False, value=value, detail=f"token f1 {value:.2f}")
