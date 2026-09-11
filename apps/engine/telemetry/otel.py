"""OpenTelemetry spans built from trace events, exported over OTLP, read by Phoenix.

Spans use OpenInference attributes, so Phoenix shows each run as a tree:

  agent.run or agent.confirm   CHAIN, one per run, with session.id and the request
    step                       AGENT, one per plan step, with the skills it equipped
      llm                      LLM, one per model call, with the prompt, the reply and tokens
      tool                     TOOL, one per tool call, with its arguments and result

A model or tool span is created when its call ends, with its start set from the duration the
event carries. A run's span stays open while it waits on a confirmation and ends when the
run's run_done event arrives.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from opentelemetry import trace as ot
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from engine.core.config import Telemetry
from engine.core.types.agent import TraceEvent, TraceKind

KIND = "openinference.span.kind"


def ns(at: datetime) -> int:
    """A timestamp in nanoseconds since the epoch."""
    return int(at.timestamp() * 1_000_000_000)


def started(event: TraceEvent) -> int:
    """When the call an event ends began, from the duration it carries."""
    duration = float(event.data.get("duration_ms", 0))
    return ns(event.at - timedelta(milliseconds=duration))


def value(raw: Any) -> str | int | float | bool:
    """An attribute value: scalars as they are, anything else as JSON."""
    if isinstance(raw, (str, int, float, bool)):
        return raw
    return json.dumps(raw, default=str)


class OtelSink:
    """Turns trace events into spans. Satisfies TraceSink."""

    def __init__(self, tracer: ot.Tracer) -> None:
        self.tracer = tracer
        self._runs: dict[str, ot.Span] = {}
        self._steps: dict[tuple[str, str], ot.Span] = {}
        self._arguments: dict[tuple[str, str, str], list[str]] = {}

    def emit(self, event: TraceEvent) -> None:
        try:
            self._emit(event)
        except Exception:  # a span that cannot be built never stops a run
            return

    def _emit(self, event: TraceEvent) -> None:
        sid = event.session_id
        if event.kind in (TraceKind.RUN_STARTED, TraceKind.CONFIRMED):
            self._close(sid, event.at, "superseded")
            name = "agent.run" if event.kind is TraceKind.RUN_STARTED else "agent.confirm"
            span = self.tracer.start_span(name, start_time=ns(event.at))
            span.set_attribute(KIND, "CHAIN")
            span.set_attribute("session.id", sid)
            span.set_attribute("input.value", value(event.data.get("request", event.data)))
            self._runs[sid] = span
            return
        root = self._runs.get(sid)
        if root is None:
            return
        if event.kind is TraceKind.RUN_DONE:
            root.set_attribute("bijou.status", str(event.data.get("status", "")))
            if "answer" in event.data:
                root.set_attribute("output.value", value(event.data["answer"]))
            self._close(sid, event.at, None)
            return
        parent = self._step(event, root) if event.step_id else root
        context = ot.set_span_in_context(parent)
        data = event.data
        if event.kind is TraceKind.MODEL_CALL:
            span = self.tracer.start_span("llm", context=context, start_time=started(event))
            span.set_attribute(KIND, "LLM")
            span.set_attribute("llm.model_name", str(data.get("model", "")))
            span.set_attribute("llm.token_count.prompt", int(data.get("prompt_tokens", 0)))
            span.set_attribute("llm.token_count.completion", int(data.get("completion_tokens", 0)))
            span.set_attribute("bijou.purpose", str(data.get("purpose", "")))
            if "input" in data:
                span.set_attribute("input.value", str(data["input"]))
                span.set_attribute("input.mime_type", "application/json")
            if "output" in data:
                span.set_attribute("output.value", str(data["output"]))
                span.set_attribute("output.mime_type", "application/json")
            span.end(ns(event.at))
        elif event.kind is TraceKind.TOOL_CALL:
            key = (sid, event.step_id, str(data.get("tool", "")))
            self._arguments.setdefault(key, []).append(str(data.get("arguments", "")))
        elif event.kind is TraceKind.TOOL_RESULT:
            tool = str(data.get("tool", ""))
            pending = self._arguments.get((sid, event.step_id, tool)) or [""]
            span = self.tracer.start_span("tool", context=context, start_time=started(event))
            span.set_attribute(KIND, "TOOL")
            span.set_attribute("tool.name", tool)
            span.set_attribute("input.value", pending.pop(0))
            span.set_attribute("output.value", str(data.get("preview", "")))
            if not data.get("ok", True):
                span.set_status(ot.Status(ot.StatusCode.ERROR))
            span.end(ns(event.at))
        elif event.kind is TraceKind.SKILLS_PICKED:
            parent.set_attribute("bijou.skills", value(data.get("skills", [])))
            parent.set_attribute("bijou.pick_reason", str(data.get("reason", "")))
        elif event.kind is TraceKind.STEP_DONE:
            parent.set_attribute("bijou.status", str(data.get("status", "")))
            parent.end(ns(event.at))
            self._steps.pop((sid, event.step_id), None)
        else:
            parent.add_event(
                event.kind.value,
                {k: value(v) for k, v in data.items()},
                timestamp=ns(event.at),
            )

    def _step(self, event: TraceEvent, root: ot.Span) -> ot.Span:
        key = (event.session_id, event.step_id)
        span = self._steps.get(key)
        if span is None:
            span = self.tracer.start_span(
                "step", context=ot.set_span_in_context(root), start_time=started(event)
            )
            span.set_attribute(KIND, "AGENT")
            span.set_attribute("bijou.step", event.step_id)
            self._steps[key] = span
        return span

    def _close(self, sid: str, at: datetime, status: str | None) -> None:
        for key in [k for k in self._steps if k[0] == sid]:
            self._steps.pop(key).end(ns(at))
        root = self._runs.pop(sid, None)
        if root is not None:
            if status is not None:
                root.set_attribute("bijou.status", status)
            root.end(ns(at))


def open_tracer(cfg: Telemetry) -> tuple[OtelSink, Callable[[], None]] | None:
    """A sink exporting to cfg.otlp_endpoint, and the call that flushes it; None when unset."""
    if not cfg.otlp_endpoint:
        return None
    provider = TracerProvider(
        resource=Resource.create({"service.name": cfg.service_name}),
        sampler=TraceIdRatioBased(cfg.sample_ratio),
    )
    exporter = OTLPSpanExporter(endpoint=cfg.otlp_endpoint, timeout=cfg.export_timeout_secs)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return OtelSink(provider.get_tracer("bijou.engine")), provider.shutdown
