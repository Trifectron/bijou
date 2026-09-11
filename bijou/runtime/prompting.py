"""The tuned-prompt baseline.

Every adapter is compared against the same base model under the best prompt found
for its skill: one of the skill's instructions and a number of worked examples,
chosen by pass rate on a dev split drawn from prompting.seed. The worked examples
are the first samples of the train split, the data an adapter learns from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import ModuleType

from bijou.backends.nanodiff import NanoDiffBackend
from bijou.core.config import Config
from bijou.core.types import AdapterState, Sample
from bijou.routing.phase import PhaseSchedule
from bijou.runtime.evaluate import score_condition
from bijou.skills import load as load_skill


@dataclass(frozen=True)
class PromptChoice:
    """An index into the skill's INSTRUCTIONS, a shot count, and its dev scores."""

    instruction: int
    shots: int
    dev_rate: float = 0.0
    dev_value: float = 0.0


def render(sample: Sample, instruction: str, demos: Sequence[Sample]) -> Sample:
    """The sample with its prompt rebuilt from an instruction and worked examples."""
    shown = "".join(f"{d.meta['text']}\n{d.target}\n\n" for d in demos)
    return replace(sample, prompt=f"{instruction}\n\n{shown}{sample.meta['text']}")


def demonstrations(cfg: Config, skill: ModuleType, shots: int) -> list[Sample]:
    """The first shots samples of the train split."""
    return list(skill.generate(shots, cfg.train.seed)) if shots else []


def candidates(cfg: Config, skill: ModuleType) -> list[PromptChoice]:
    """Every instruction crossed with every configured shot count."""
    return [
        PromptChoice(instruction=i, shots=k)
        for i in range(len(skill.INSTRUCTIONS))
        for k in cfg.prompting.shots
    ]


def prompted(
    cfg: Config, skill: ModuleType, choice: PromptChoice, samples: Sequence[Sample]
) -> list[Sample]:
    """The samples rendered under one prompt choice."""
    demos = demonstrations(cfg, skill, choice.shots)
    instruction = skill.INSTRUCTIONS[choice.instruction]
    return [render(s, instruction, demos) for s in samples]


def tune(
    cfg: Config, backend: NanoDiffBackend, state: AdapterState, skill_name: str
) -> PromptChoice:
    """The candidate with the best dev pass rate, then mean value, then fewest shots."""
    skill = load_skill(skill_name)
    dev = skill.generate(cfg.prompting.dev_samples, cfg.prompting.seed)
    scored = []
    for choice in candidates(cfg, skill):
        report, _ = score_condition(
            cfg,
            backend,
            state,
            skill_name,
            "prompt-dev",
            PhaseSchedule.static(),
            samples=prompted(cfg, skill, choice, dev),
        )
        scored.append(replace(choice, dev_rate=report.rate, dev_value=report.mean_value))
    return max(scored, key=lambda c: (c.dev_rate, c.dev_value, -c.shots, -c.instruction))


def tuned_eval_split(cfg: Config, skill_name: str, choice: PromptChoice) -> list[Sample]:
    """The eval split rendered under the tuned prompt."""
    skill = load_skill(skill_name)
    return prompted(cfg, skill, choice, skill.generate(cfg.eval.eval_samples, cfg.eval.seed))
