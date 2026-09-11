"""The teacher: an OpenAI-compatible chat model that writes a skill's examples.

The only module in bijou that opens a network connection. Structured output is requested with a
JSON schema, and a reply that does not match it is dropped rather than repaired.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import httpx

from engine.collect.spec import Pair, SkillSpec
from engine.core.config import Collect
from engine.core.types.errors import EngineError

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"input": {"type": "string"}, "output": {"type": "string"}},
                "required": ["input", "output"],
            },
        }
    },
    "required": ["examples"],
}

SYSTEM = (
    "You write training data for one narrow skill of a small language model. Every example is "
    "an input the skill will receive and the exact output it should produce. Outputs follow the "
    "instruction precisely and contain nothing else. Vary the inputs: wording, length, entities "
    "and edge cases. Never repeat an input you were shown."
)


class TeacherError(EngineError):
    """The teacher could not be reached or answered with something unusable."""


class Teacher(Protocol):
    """Writes up to n new examples for a spec, given examples already written."""

    def write(self, spec: SkillSpec, n: int, seen: list[Pair]) -> list[Pair]: ...


def prompt(spec: SkillSpec, n: int, seen: list[Pair]) -> str:
    """The user message for one request."""
    lines = [
        f"Skill: {spec.name}",
        f"What it does: {spec.description}",
        f"Instruction every input is given with: {spec.instruction}",
    ]
    if spec.examples:
        lines.append("Inputs seen in practice:")
        lines += [f"- {e}" for e in spec.examples[:20]]
    shown = (spec.pairs + seen)[-8:]
    if shown:
        lines.append("Examples already written:")
        lines += [json.dumps(p.model_dump()) for p in shown]
    lines.append(f'Write {n} new examples as JSON: {{"examples": [{{"input", "output"}}]}}.')
    return "\n".join(lines)


class OpenAITeacher:
    """A Teacher over an OpenAI-compatible /chat/completions endpoint. Satisfies Teacher."""

    def __init__(self, cfg: Collect, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        headers = {}
        if cfg.api_key.get_secret_value():
            headers["Authorization"] = f"Bearer {cfg.api_key.get_secret_value()}"
        self.client = client or httpx.Client(
            base_url=cfg.base_url.rstrip("/"), timeout=cfg.timeout_secs, headers=headers
        )

    def write(self, spec: SkillSpec, n: int, seen: list[Pair]) -> list[Pair]:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt(spec, n, seen)},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "examples", "schema": SCHEMA},
            },
            "chat_template_kwargs": {"enable_thinking": self.cfg.thinking},
        }
        try:
            response = self.client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise TeacherError(f"teacher at {self.cfg.base_url} unreachable: {exc}") from exc
        if response.status_code != 200:
            raise TeacherError(f"teacher returned {response.status_code}: {response.text[:200]}")
        return parse(response.json())


def parse(payload: dict[str, Any]) -> list[Pair]:
    """The examples in one chat completion. Malformed items are dropped."""
    try:
        content = payload["choices"][0]["message"]["content"] or ""
        items = json.loads(content)["examples"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return []
    pairs = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and str(item.get("input", "")).strip():
            output = str(item.get("output", "")).strip()
            if output:
                pairs.append(Pair(input=str(item["input"]).strip(), output=output))
    return pairs
