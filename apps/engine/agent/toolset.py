"""The tools one subagent can see, by name."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from engine.core.protocols import Tool
from engine.core.types.agent import ToolDefinition
from engine.core.types.errors import ConfigError


class ToolSet:
    """An immutable, name-keyed set of tools. Two tools cannot share a name."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            name = tool.definition.name
            if name in self._tools:
                raise ConfigError(f"two tools are named {name}")
            self._tools[name] = tool

    def __len__(self) -> int:
        return len(self._tools)

    def plus(self, *tools: Tool) -> ToolSet:
        """A new set with these tools added."""
        return ToolSet([*self._tools.values(), *tools])

    def without(self, names: Iterable[str]) -> ToolSet:
        """A new set with the named tools removed."""
        dropped = set(names)
        return ToolSet([t for n, t in self._tools.items() if n not in dropped])

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def definitions(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]
