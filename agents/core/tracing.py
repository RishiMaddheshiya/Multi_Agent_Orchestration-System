"""Execution tracing and per-run cost/performance metrics.

Each graph node creates a `NodeTrace`, which records a root "node" event plus child events
(LLM calls, tool calls, memory operations, human decisions, retries, errors). The node
returns `trace.events()` into the LangGraph state, where a list reducer appends them. Since
the trace lives in graph state, it survives human-in-the-loop pauses and is persisted
with the run.
"""

from __future__ import annotations

import json
import time
from typing import Any

from core.config import get_logger, get_settings
from core.schemas import (
    HumanDecision,
    LLMCallInfo,
    RunMetrics,
    TokenUsage,
    ToolCallRecord,
    TraceEvent,
)

log = get_logger("trace")


def _short(value: Any, limit: int = 1200) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + " …"


class NodeTrace:
    def __init__(self, trace_id: str, node: str, agent: str = "system", *, input: Any = None):
        self.trace_id = trace_id
        self.node = node
        self.agent = agent
        self._started = time.perf_counter()
        self.root = TraceEvent(
            trace_id=trace_id, node=node, agent=agent, event_type="node", name=node, input=_short(input)
        )
        self._children: list[TraceEvent] = []

    # -- child events ------------------------------------------------------------------
    def llm(self, name: str, info: LLMCallInfo, *, input: Any = None, output: Any = None,
            agent: str | None = None, confidence: float | None = None) -> TraceEvent:
        event = TraceEvent(
            trace_id=self.trace_id, parent_id=self.root.event_id, node=self.node, agent=agent or self.agent,
            event_type="llm_call", name=name, input=_short(input), output=_short(output),
            latency_ms=round(info.latency_ms, 1), token_usage=info.usage, llm_calls=1, model=info.model,
            confidence=confidence,
        )
        if info.attempts > 1:
            event.error = f"succeeded after {info.attempts} attempts"
        self._children.append(event)
        return event

    def tool(self, record: ToolCallRecord) -> TraceEvent:
        status = {"success": "success", "human_provided": "success", "denied": "skipped"}.get(record.status, "error")
        event = TraceEvent(
            trace_id=self.trace_id, parent_id=self.root.event_id, node=self.node, agent=record.agent,
            event_type="tool_call", name=record.tool, tool=record.tool, tool_input=record.input,
            tool_output=_short(record.output, 2000), latency_ms=round(record.latency_ms, 1),
            token_usage=record.token_usage if record.token_usage.total_tokens else None, llm_calls=record.llm_calls,
            status=status, error=record.error,
            human_decision=None if record.approved_by_human is None else ("approved" if record.approved_by_human else "denied"),
        )
        self._children.append(event)
        return event

    def memory(self, name: str, *, input: Any = None, output: Any = None, latency_ms: float = 0.0,
               status: str = "success", error: str | None = None) -> TraceEvent:
        event = TraceEvent(
            trace_id=self.trace_id, parent_id=self.root.event_id, node=self.node, agent=self.agent,
            event_type="memory", name=name, input=_short(input), output=_short(output),
            latency_ms=round(latency_ms, 1), status=status, error=error,  # type: ignore[arg-type]
        )
        self._children.append(event)
        return event

    def human(self, decision: HumanDecision) -> TraceEvent:
        event = TraceEvent(
            trace_id=self.trace_id, parent_id=self.root.event_id, node=self.node, agent="human",
            event_type="human", name=f"human_{decision.checkpoint}", input=_short(decision.subject),
            output=_short(decision.modified_instruction or decision.feedback), human_decision=decision.decision,
            latency_ms=round(decision.review_seconds * 1000, 1),
        )
        self._children.append(event)
        return event

    def info(self, name: str, message: Any, *, event_type: str = "info", status: str = "info",
             error: str | None = None) -> TraceEvent:
        event = TraceEvent(
            trace_id=self.trace_id, parent_id=self.root.event_id, node=self.node, agent=self.agent,
            event_type=event_type, name=name, output=_short(message), status=status, error=error,  # type: ignore[arg-type]
        )
        self._children.append(event)
        return event

    def error(self, name: str, exc: Exception | str, *, user_message: str | None = None) -> TraceEvent:
        detail = str(exc)
        log.error("[%s/%s] %s: %s", self.node, self.agent, name, detail)
        return self.info(name, user_message or detail, event_type="error", status="error", error=detail[:1000])

    def retry(self, name: str, attempt: int, error: str) -> TraceEvent:
        return self.info(name, f"Retry {attempt}: {error}", event_type="retry", status="error", error=error[:1000])

    # -- finish ------------------------------------------------------------------------
    def events(self, *, status: str = "success", output: Any = None, error: str | None = None,
               confidence: float | None = None) -> list[TraceEvent]:
        self.root.latency_ms = round((time.perf_counter() - self._started) * 1000, 1)
        self.root.status = status  # type: ignore[assignment]
        self.root.output = _short(output)
        self.root.error = error
        self.root.confidence = confidence
        usage = TokenUsage()
        calls = 0
        for child in self._children:
            if child.token_usage:
                usage = usage + child.token_usage
            calls += child.llm_calls
        self.root.token_usage = usage if usage.total_tokens else None
        self.root.model = get_settings().gemini_model if calls else None
        return [self.root, *self._children]


def compute_metrics(trace: list[TraceEvent], human_decisions: list[HumanDecision], execution_time_s: float) -> RunMetrics:
    settings = get_settings()
    usage = TokenUsage()
    llm_calls = tool_calls = 0
    for event in trace:
        if event.event_type == "node":
            continue  # node roots aggregate their children; avoid double counting
        if event.token_usage:
            usage = usage + event.token_usage
        llm_calls += event.llm_calls
        if event.event_type == "tool_call" and event.status in ("success", "error"):
            tool_calls += 1
    human_time = sum(d.review_seconds for d in human_decisions)
    cost = None
    label = "Cost unavailable"
    if settings.pricing_configured:
        cost = (usage.input_tokens * settings.input_price_per_million
                + usage.output_tokens * settings.output_price_per_million) / 1_000_000
        label = f"${cost:.4f}"
    return RunMetrics(
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, total_tokens=usage.total_tokens,
        llm_calls=llm_calls, tool_calls=tool_calls, execution_time_s=round(execution_time_s, 2),
        human_review_time_s=round(human_time, 1),
        # Plan/action approvals are opt-in gates; escalations are the system asking for help.
        human_escalations=sum(1 for d in human_decisions if d.checkpoint in ("final_output", "failure")),
        cost_usd=cost, cost_label=label,
    )
