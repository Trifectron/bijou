"""Skill: free text to a fixed JSON schema.

Chosen as skill zero because it is synthetic, so data is unlimited and
unlicensed; auto-gradable, so no judge model sits between the adapter and the
number; and a format task rather than a knowledge task, so a 150M model can
plausibly learn it.
"""

from __future__ import annotations

import json
import random

from bijou.core.types import Sample, Score

NAME = "json_extract"
FIELDS = ("name", "role", "city", "years")
ROLES = ("engineer", "analyst", "designer", "researcher", "manager")
CITIES = ("Tempe", "Seattle", "Austin", "Boston", "Denver")
NAMES = ("Ana", "Ben", "Chen", "Dara", "Eli", "Farah", "Gita", "Hugo")

# Candidates for the tuned-prompt baseline. The first is the default prompt.
INSTRUCTIONS = (
    "Extract name, role, city and years from the text as JSON.",
    'Reply with one JSON object with the keys "name", "role", "city" and "years" '
    "for the person described below. years is a number.",
)
PROMPT = "{instruction}\n\n{text}"


def generate(n: int, seed: int) -> list[Sample]:
    """n samples drawn from seed. Satisfies SkillData."""
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        record = {
            "name": rng.choice(NAMES),
            "role": rng.choice(ROLES),
            "city": rng.choice(CITIES),
            "years": rng.randint(1, 30),
        }
        text = (
            f"{record['name']} has worked as a {record['role']} in "
            f"{record['city']} for {record['years']} years."
        )
        samples.append(
            Sample(
                id=f"{NAME}-{seed}-{i}",
                prompt=PROMPT.format(instruction=INSTRUCTIONS[0], text=text),
                target=json.dumps(record),
                meta={"text": text},
            )
        )
    return samples


def grade(sample: Sample, output: str) -> Score:
    """Parse the output and compare every field. Satisfies Grader."""
    try:
        got = json.loads(output.strip())
    except json.JSONDecodeError as exc:
        return Score(passed=False, value=0.0, detail=f"unparseable: {exc.msg}")
    if not isinstance(got, dict):
        return Score(passed=False, value=0.0, detail=f"expected object, got {type(got).__name__}")
    want = json.loads(sample.target)
    correct = sum(1 for f in FIELDS if str(got.get(f, "")) == str(want[f]))
    missing = [f for f in FIELDS if f not in got]
    detail = "" if correct == len(FIELDS) else f"missing {missing}, {correct}/{len(FIELDS)} fields"
    return Score(passed=correct == len(FIELDS), value=correct / len(FIELDS), detail=detail)
