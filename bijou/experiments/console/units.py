"""The catalog of everything the console can run, in sidebar order, and its command line.

Every unit is a just recipe line. Each skill adds a train and a full fine-tune unit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class Group(StrEnum):
    """A sidebar section."""

    SETUP = "setup"
    GATE = "gate"
    TRAIN = "train"
    EVALUATE = "evaluate"
    INSPECT = "inspect"
    DEPLOY = "deploy"
    ADHOC = "ad-hoc"


@dataclass(frozen=True)
class Unit:
    """One just recipe line. Its id is the line itself."""

    group: Group
    args: tuple[str, ...]
    hint: str

    @property
    def id(self) -> str:
        return " ".join(self.args)


def _unit(group: Group, line: str, hint: str) -> Unit:
    return Unit(group, tuple(line.split()), hint)


def adhoc(args: Sequence[str]) -> Unit:
    """A recipe typed at the command line."""
    return Unit(Group.ADHOC, tuple(args), "typed at the command line")


def catalog(skills: Sequence[str]) -> list[Unit]:
    """Every unit the console lists, for the given skills."""
    units = [
        _unit(Group.SETUP, "doctor", "tools, submodule, dependencies, checkpoint, GPU"),
        _unit(Group.SETUP, "setup", "dependencies without torch"),
        _unit(Group.SETUP, "setup-train", "dependencies with CUDA torch"),
        _unit(Group.SETUP, "setup-train-cpu", "dependencies with CPU torch"),
        _unit(Group.SETUP, "vendor", "fetch the nanoDiff submodule"),
        _unit(Group.SETUP, "checkpoints", "download the configured base checkpoint"),
        _unit(Group.GATE, "check", "format, lint, layering, types, tests"),
        _unit(Group.GATE, "fmt", "format in place"),
        _unit(Group.GATE, "check-model", "the tests with the model stack, none skipped"),
        _unit(Group.GATE, "test-gpu", "the base checkpoint on the configured device"),
    ]
    for skill in skills:
        units.append(_unit(Group.TRAIN, f"skill train {skill}", f"LoRA adapter on {skill}"))
        units.append(
            _unit(
                Group.TRAIN,
                f"skill train {skill} --full-finetune",
                f"full fine-tune on {skill}, the upper bound",
            )
        )
    units += [
        _unit(Group.EVALUATE, "evaluate", "score the composition matrix"),
        _unit(Group.EVALUATE, "matrix", "train everything, then score the matrix"),
        _unit(Group.INSPECT, "runs", "every run record"),
        _unit(Group.INSPECT, "config", "the resolved configuration"),
        _unit(Group.INSPECT, "skill list", "skills and what is trained"),
        _unit(Group.DEPLOY, "image", "build the CUDA training image"),
        _unit(Group.DEPLOY, "image cpu", "build the CPU training image"),
    ]
    return units


Verb = Literal["quit", "help", "clear", "start", "stop", "restart", "just", "none"]


@dataclass(frozen=True)
class Command:
    """A parsed command line. arg names the unit for start, stop and restart."""

    verb: Verb
    arg: str = ""
    args: tuple[str, ...] = ()


def parse_command(text: str) -> Command:
    """The command typed after a colon. Anything unrecognised runs as a just recipe."""
    words = text.split()
    if not words:
        return Command("none")
    head, rest = words[0], words[1:]
    arg = " ".join(rest)
    if head in ("q", "quit"):
        return Command("quit")
    if head in ("h", "help"):
        return Command("help")
    if head == "clear":
        return Command("clear")
    if head == "start" and arg:
        return Command("start", arg=arg)
    if head == "stop" and arg:
        return Command("stop", arg=arg)
    if head == "restart" and arg:
        return Command("restart", arg=arg)
    if head == "just":
        return Command("just", args=tuple(rest))
    return Command("just", args=tuple(words))
