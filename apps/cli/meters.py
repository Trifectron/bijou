"""The metrics pane: what the agent has counted since it started.

The agent answers {"op": "stats"} with every sample in its Prometheus registry. Counters are
totals, so a rate is the difference between two looks; each histogram carries a _sum and a
_count, whose ratio is the mean. Nothing is kept on disk: the pane shows this agent's run, and
Grafana keeps the history.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rich.text import Text

BARS = "▁▂▃▄▅▆▇█"
# Looks kept for the sparklines, about four minutes at the default interval.
HISTORY = 120
# The counters whose change between looks is worth a sparkline.
RATES = ("bijou_agent_runs_total", "bijou_agent_tokens_total", "bijou_agent_tool_calls_total")


def spark(values: Sequence[float], width: int) -> str:
    """The last width values as one line of bars, scaled to the largest."""
    window = [v for v in values][-max(width, 1) :]
    if not window:
        return ""
    high = max(window)
    if high <= 0:
        return BARS[0] * len(window)
    return "".join(BARS[min(int(v / high * (len(BARS) - 1) + 0.5), len(BARS) - 1)] for v in window)


def _count(value: float) -> str:
    """A count as people read it: 7, 1.2k, 3.4M."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def _breakdown(parts: Mapping[str, float], limit: int = 4) -> str:
    """The largest few label values, as name n, joined by dots."""
    ranked = sorted(((v, k) for k, v in parts.items() if v), reverse=True)[:limit]
    return " · ".join(f"{name} {_count(value)}" for value, name in ranked)


@dataclass
class Meters:
    """The agent's counters as last seen, and how fast the interesting ones move."""

    latest: dict[str, float] = field(default_factory=dict)
    rates: dict[str, deque[float]] = field(default_factory=dict)

    def update(self, stats: Mapping[str, float]) -> None:
        """Take one look. The first fills the totals; later ones also feed the sparklines."""
        for name in RATES:
            was = self.total(name) if self.latest else None
            now = sum(v for k, v in stats.items() if _is(k, name))
            if was is not None:
                self.rates.setdefault(name, deque(maxlen=HISTORY)).append(max(now - was, 0.0))
        self.latest = dict(stats)

    def total(self, name: str) -> float:
        """One metric's value, every label summed."""
        return sum(value for key, value in self.latest.items() if _is(key, name))

    def split(self, name: str, label: str) -> dict[str, float]:
        """One metric's values, grouped by one label."""
        out: dict[str, float] = {}
        for key, value in self.latest.items():
            if not key.startswith(f"{name}{{"):
                continue
            for part in key[len(name) + 1 : -1].split(","):
                found, _, found_value = part.partition("=")
                if found == label:
                    out[found_value] = out.get(found_value, 0.0) + value
        return out

    def mean(self, histogram: str) -> str:
        """A histogram's mean, or an empty string before it has anything in it."""
        count = self.total(f"{histogram}_count")
        return f"{self.total(f'{histogram}_sum') / count:.1f}s" if count else ""

    def render(self, width: int, machine: Sequence[str] = ()) -> Text:
        """One line per thing the agent does, with a sparkline where a rate means something.

        machine holds the GPU and box rows, shown down the right of the pane.
        """
        if not self.latest:
            return Text("waiting for the agent", style="dim")
        column = max(width - 34, 30) if machine else width
        bars = max(min(column - 52, 40), 8)
        out = Text(no_wrap=True, overflow="ellipsis")
        runs, tokens = (
            self.split("bijou_agent_runs_total", "status"),
            self.split("bijou_agent_tokens_total", "kind"),
        )
        calls = self.split("bijou_agent_model_calls_total", "purpose")
        errors = self.split("bijou_agent_model_calls_total", "outcome").get("error", 0.0)
        tools = self.split("bijou_agent_tool_calls_total", "tool")
        rows = [
            ("runs", _count(sum(runs.values())), _breakdown(runs), self._spark("runs", bars)),
            ("steps", _count(self.total("bijou_agent_steps_total")), "", ""),
            (
                "model",
                _count(sum(calls.values())),
                _breakdown(calls) + (f" · {_count(errors)} failed" if errors else ""),
                self.mean("bijou_agent_model_call_seconds"),
            ),
            (
                "tokens",
                _count(sum(tokens.values())),
                _breakdown(tokens),
                self._spark("tokens", bars),
            ),
            (
                "tools",
                _count(sum(tools.values())),
                _breakdown(tools),
                self.mean("bijou_agent_tool_seconds"),
            ),
            (
                "policy",
                "",
                _breakdown(self.split("bijou_agent_policy_decisions_total", "decision")),
                "",
            ),
            ("skills", "", _breakdown(self.split("bijou_agent_skill_picks_total", "skills")), ""),
            (
                "bank",
                _count(self.total("bijou_bank_generations_total")),
                f"{_count(self.total('bijou_bank_trained_skills'))} trained",
                self.mean("bijou_bank_generation_seconds"),
            ),
        ]
        for index, (name, total, detail, tail) in enumerate(rows):
            if out.plain:
                out.append("\n")
            line = Text.assemble((f"{name:<7}", "bold"), f"{total:>6}  ")
            line.append(f"{detail:<34}", "dim" if not detail else "")
            line.append(tail, "cyan" if tail and tail[0] in BARS else "dim")
            line.truncate(column - 2, pad=True)
            out.append_text(line)
            if index < len(machine):
                out.append(machine[index], "dim")
        return out

    def _spark(self, name: str, width: int) -> str:
        return spark(self.rates.get(f"bijou_agent_{name}_total", ()), width)


def _is(key: str, name: str) -> bool:
    return key == name or key.startswith(f"{name}{{")
