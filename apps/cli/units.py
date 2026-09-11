"""The catalog of everything the console runs, in sidebar order, and its command line.

Every unit is a just recipe line, and each skill adds a train unit. Most units are tasks, run when
asked. The agent starts with the console and speaks JSON lines to the chat pane; a follow unit
tails one compose service's logs and starts when that service comes up; an interactive unit,
such as nvtop, is handed the terminal instead of streaming into the log pane.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class Group(StrEnum):
    """A sidebar section."""

    SERVICES = "services"
    SETUP = "setup"
    GATE = "gate"
    STACK = "stack"
    SKILLS = "skills"
    EVALUATE = "evaluate"
    INSPECT = "inspect"
    MONITOR = "monitor"
    DEPLOY = "deploy"
    ADHOC = "ad-hoc"


class Kind(StrEnum):
    """How the console runs a unit."""

    TASK = "task"
    AGENT = "agent"
    FOLLOW = "follow"
    INTERACTIVE = "interactive"


@dataclass(frozen=True)
class Unit:
    """One just recipe line. Its id is the line itself; name is what the sidebar shows."""

    group: Group
    args: tuple[str, ...]
    hint: str
    kind: Kind = Kind.TASK
    label: str = ""
    # The compose service a follow unit tails.
    service: str = ""

    @property
    def id(self) -> str:
        return " ".join(self.args)

    @property
    def name(self) -> str:
        return self.label or self.id


def _unit(
    group: Group, line: str, hint: str, kind: Kind = Kind.TASK, label: str = "", service: str = ""
) -> Unit:
    return Unit(group, tuple(line.split()), hint, kind, label, service)


def adhoc(args: Sequence[str]) -> Unit:
    """A recipe typed at the command line."""
    return Unit(Group.ADHOC, tuple(args), "typed at the command line")


# The compose services whose logs the console follows: name in deploy/compose.yml, label, hint.
SERVICES = (
    ("chat", "llama-server", "the chat model"),
    ("phoenix", "phoenix", "traces of every run"),
    ("prometheus", "prometheus", "metrics"),
    ("grafana", "grafana", "dashboards"),
    ("gpu-exporter", "gpu-exporter", "GPU metrics for prometheus"),
)


def catalog(skills: Sequence[str]) -> list[Unit]:
    """Every unit the console lists, for the given skills."""
    units = [
        _unit(Group.SERVICES, "chat --jsonl", "chat with it on the right", Kind.AGENT, "agent"),
    ]
    units += [
        _unit(Group.SERVICES, f"logs {service}", hint, Kind.FOLLOW, label, service)
        for service, label, hint in SERVICES
    ]
    units += [
        _unit(Group.SETUP, "doctor", "tools, submodule, dependencies, checkpoint, services"),
        _unit(Group.SETUP, "setup", "every app, no torch"),
        _unit(Group.SETUP, "setup cuda", "every app, with CUDA torch"),
        _unit(Group.SETUP, "checkpoints", "download the configured base checkpoint"),
        _unit(Group.GATE, "check", "format, lint, layering, types, tests"),
        _unit(Group.GATE, "fmt", "format in place"),
        _unit(Group.GATE, "test model", "the tests with torch, none skipped"),
        _unit(Group.GATE, "test gpu", "the base checkpoint on the configured device"),
        _unit(Group.STACK, "up model", "llama-server, the chat model"),
        _unit(Group.STACK, "up observe", "phoenix, prometheus and grafana"),
        _unit(Group.STACK, "up gpu", "the GPU exporter"),
        _unit(Group.STACK, "down", "stop every compose service"),
        _unit(Group.SKILLS, "skills", "every skill and what is trained"),
        _unit(Group.SKILLS, "skills propose", "recurring work no skill covers"),
        _unit(Group.SKILLS, "skills specs", "skill specs waiting for review"),
    ]
    units += [
        _unit(Group.SKILLS, f"skills train {skill}", f"LoRA adapter on {skill}") for skill in skills
    ]
    units += [
        _unit(Group.EVALUATE, "evals", "golden cases through the engine, gated on the baseline"),
        _unit(Group.EVALUATE, "matrix", "score the composition matrix"),
        _unit(Group.EVALUATE, "matrix --train", "train everything, then score the matrix"),
        _unit(Group.INSPECT, "sessions", "every session, newest first"),
        _unit(Group.INSPECT, "runs", "every run record"),
        _unit(Group.INSPECT, "config", "the resolved configuration"),
        _unit(Group.MONITOR, "nvtop", "the GPU, full screen", Kind.INTERACTIVE),
        _unit(Group.MONITOR, "htop", "processes and CPU, full screen", Kind.INTERACTIVE),
        _unit(Group.DEPLOY, "image", "build the engine image"),
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
