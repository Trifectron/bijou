"""The pattern miner: recurring work that no skill covered, proposed as new skills.

A step that ran with nothing equipped is work the LoRA bank does not cover. Such steps are
clustered by the keywords of their goals; a cluster seen in patterns.min_occurrences distinct
sessions becomes a proposal. Answered steps become worked pairs, so the teacher that collects
the skill's data starts from what the harness actually did.

The gate is rules, not a model call, so mining is deterministic and free. Every proposal is
written with approved false; bijou collect refuses it until a person has read and approved it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from engine.core.config import Patterns
from engine.core.protocols import SessionStore
from engine.core.types.agent import RunStatus, SkillPair, SkillProposal, StepReport

_STOPWORD_TEXT = """
    the and for with that this from into onto about over under your you our their them they what
    when where which while who whom whose why how all any are was were been being have has had
    does did doing done can could should would will shall may might must not but than then also
    just only very more most some such each every other its it's his her him she himself find
    get give make show tell list look check keep need want use using please new latest
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def keywords(text: str) -> frozenset[str]:
    """The content words of a goal: lower case, three letters or more, no stopwords."""
    return frozenset(
        w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOPWORDS
    )


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def slug(words: list[str]) -> str:
    name = "_".join(words)[:64] or "skill"
    return name if name[0].isalpha() else f"skill_{name}"[:64]


@dataclass
class Cluster:
    """Steps whose goals share keywords with the first one seen."""

    seed: frozenset[str]
    items: list[tuple[str, StepReport]] = field(default_factory=list)

    @property
    def sessions(self) -> list[str]:
        return sorted({session for session, _ in self.items})


class PatternMiner:
    """Reads sessions and proposes skills."""

    def __init__(self, sessions: SessionStore, cfg: Patterns) -> None:
        self.sessions = sessions
        self.cfg = cfg

    def uncovered(self) -> list[tuple[str, StepReport]]:
        """Every step in the window that ran with no skill equipped, with its session id."""
        found = []
        for record in self.sessions.records(self.cfg.window):
            found += [(record.id, step) for step in record.steps if not step.skills]
        return found

    def clusters(self) -> list[Cluster]:
        clusters: list[Cluster] = []
        for session, step in self.uncovered():
            words = keywords(step.goal)
            if not words:
                continue
            best = max(clusters, key=lambda c: jaccard(words, c.seed), default=None)
            if best is not None and jaccard(words, best.seed) >= self.cfg.similarity:
                best.items.append((session, step))
            else:
                clusters.append(Cluster(seed=words, items=[(session, step)]))
        return clusters

    def propose(self, existing: set[str] | None = None) -> list[SkillProposal]:
        """One proposal per cluster seen in enough sessions, most frequent first."""
        taken = set(existing or ())
        proposals = []
        for cluster in self.clusters():
            if len(cluster.sessions) < self.cfg.min_occurrences:
                continue
            counts = Counter(w for _, step in cluster.items for w in keywords(step.goal))
            top = [w for w, _ in counts.most_common(3)]
            name, n = slug(top), 2
            while name in taken:
                name, n = f"{slug(top)[:60]}_{n}", n + 1
            taken.add(name)
            goals = list(dict.fromkeys(step.goal for _, step in cluster.items))
            representative = max(goals, key=lambda g: len(keywords(g) & set(top)))
            pairs = [
                SkillPair(input=step.goal, output=step.answer)
                for _, step in cluster.items
                if step.status is RunStatus.ANSWERED and step.answer.strip()
            ]
            proposals.append(
                SkillProposal(
                    name=name,
                    description=f"Handles work like: {representative}"[:300],
                    instruction="Carry out this task and reply with the result only.",
                    examples=goals[: self.cfg.max_examples],
                    pairs=pairs[: self.cfg.max_examples],
                    occurrences=len(cluster.items),
                    sessions=cluster.sessions,
                )
            )
        return sorted(proposals, key=lambda p: p.occurrences, reverse=True)


def write_proposals(
    proposals: list[SkillProposal], directory: Path
) -> tuple[list[Path], list[Path]]:
    """Write each proposal as name.json. An existing file is left alone and reported."""
    directory.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    for proposal in proposals:
        path = directory / f"{proposal.name}.json"
        if path.exists():
            skipped.append(path)
            continue
        path.write_text(json.dumps(proposal.model_dump(), indent=2) + "\n")
        written.append(path)
    return written, skipped
