"""Short-term memory: the LangGraph state for the current task.

Everything an agent knows about the current run lives here: the task, plan, agent outputs,
tool results, review feedback and human decisions. It is checkpointed by LangGraph (which is
what makes human-in-the-loop pauses possible) and is reduced to a `RunSnapshot` for
persistence and replay.

Reducers:
  * lists of events (trace, tool calls, human decisions, notifications) are appended to;
  * dicts keyed by subtask id (subtasks, results, tool_plans) are merged, so parallel
    specialist branches can update different subtasks in the same step;
  * nodes that need to replace a dict wholesale (a new plan) return `Overwrite(...)`.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from core.schemas import (
    AgentResult,
    ConfidenceBreakdown,
    FinalResult,
    HumanDecision,
    IntakeAnalysis,
    MemoryRecord,
    RunMetrics,
    RunOptions,
    ReviewResult,
    SubTask,
    SubtaskToolPlan,
    SupervisorPlan,
    Task,
    ToolCallRecord,
    TraceEvent,
    UploadedFile,
)


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class AgentState(TypedDict, total=False):
    # task & inputs
    task: Task
    options: RunOptions
    uploads: list[UploadedFile]
    started_at: float
    # intake & memory
    intake: IntakeAnalysis | None
    retrieved_memories: list[MemoryRecord]
    memories_used: list[str]
    # planning
    raw_plan: SupervisorPlan | None
    plan_reasoning: str
    plan_version: int
    plan_status: str  # pending | approved | modify | rejected | taken_over
    plan_feedback: str | None
    plan_history: Annotated[list[list[SubTask]], operator.add]
    plan_issues: list[str]
    subtasks: Annotated[dict[str, SubTask], merge_dicts]
    # execution
    wave: list[str]
    current_subtask_id: str  # only set in Send payloads for parallel branches
    tool_plans: Annotated[dict[str, SubtaskToolPlan], merge_dicts]
    results: Annotated[dict[str, AgentResult], merge_dicts]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    escalations: list[str]
    # review & confidence
    review: ReviewResult | None
    review_history: Annotated[list[ReviewResult], operator.add]
    revision_count: int
    confidence: ConfidenceBreakdown | None
    route: str
    # human-in-the-loop
    human_decisions: Annotated[list[HumanDecision], operator.add]
    human_final_override: str | None
    human_notes: Annotated[list[str], operator.add]
    # output & observability
    final: FinalResult | None
    final_summary: str
    notifications: Annotated[list[str], operator.add]
    trace: Annotated[list[TraceEvent], operator.add]
    stored_memories: list[MemoryRecord]
    metrics: RunMetrics | None


class RunSnapshot(BaseModel):
    """Serializable view of a run, used by the UI, SQLite persistence and replay."""

    task: Task
    options: RunOptions = Field(default_factory=RunOptions)
    uploads: list[UploadedFile] = Field(default_factory=list)
    intake: IntakeAnalysis | None = None
    retrieved_memories: list[MemoryRecord] = Field(default_factory=list)
    memories_used: list[str] = Field(default_factory=list)
    plan_reasoning: str = ""
    plan_history: list[list[SubTask]] = Field(default_factory=list)
    plan_issues: list[str] = Field(default_factory=list)
    subtasks: list[SubTask] = Field(default_factory=list)
    tool_plans: list[SubtaskToolPlan] = Field(default_factory=list)
    results: dict[str, AgentResult] = Field(default_factory=dict)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    review: ReviewResult | None = None
    review_history: list[ReviewResult] = Field(default_factory=list)
    revision_count: int = 0
    confidence: ConfidenceBreakdown | None = None
    human_decisions: list[HumanDecision] = Field(default_factory=list)
    final: FinalResult | None = None
    notifications: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    stored_memories: list[MemoryRecord] = Field(default_factory=list)
    metrics: RunMetrics | None = None


def snapshot_from_state(values: dict[str, Any]) -> RunSnapshot:
    subtasks = values.get("subtasks") or {}
    tool_plans = values.get("tool_plans") or {}
    return RunSnapshot(
        task=values["task"],
        options=values.get("options") or RunOptions(),
        uploads=values.get("uploads") or [],
        intake=values.get("intake"),
        retrieved_memories=values.get("retrieved_memories") or [],
        memories_used=values.get("memories_used") or [],
        plan_reasoning=values.get("plan_reasoning") or "",
        plan_history=values.get("plan_history") or [],
        plan_issues=values.get("plan_issues") or [],
        subtasks=list(subtasks.values()),
        tool_plans=list(tool_plans.values()),
        results=values.get("results") or {},
        tool_calls=values.get("tool_calls") or [],
        review=values.get("review"),
        review_history=values.get("review_history") or [],
        revision_count=values.get("revision_count") or 0,
        confidence=values.get("confidence"),
        human_decisions=values.get("human_decisions") or [],
        final=values.get("final"),
        notifications=values.get("notifications") or [],
        trace=values.get("trace") or [],
        stored_memories=values.get("stored_memories") or [],
        metrics=values.get("metrics"),
    )
