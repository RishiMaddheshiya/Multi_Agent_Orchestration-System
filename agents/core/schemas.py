"""Typed contracts shared by agents, tools, the LangGraph workflow and the UI.

Two groups of models live here:

* Domain models (Task, SubTask, AgentResult, ReviewResult, HumanDecision, FinalResult, ...)
  are what flows through the LangGraph state and gets persisted.
* LLM output models (IntakeAnalysis, SupervisorPlan, ToolPlan, SpecialistOutput, ...) are the
  JSON schemas Gemini is constrained to. They are intentionally flat and unconstrained so the
  model can always satisfy them; values are clamped/validated after parsing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

AgentName = Literal["researcher", "analyst", "writer"]
SPECIALISTS: tuple[str, ...] = ("researcher", "analyst", "writer")
AGENT_LABELS = {
    "supervisor": "Supervisor",
    "researcher": "Research Agent",
    "analyst": "Analysis Agent",
    "writer": "Writer Agent",
    "reviewer": "Reviewer",
    "human": "Human",
    "system": "System",
}

TaskStatus = Literal["running", "awaiting_human", "completed", "rejected", "failed"]
SubTaskStatus = Literal["pending", "running", "completed", "failed", "skipped"]
DecisionType = Literal["approve", "modify", "reject", "take_over"]
ApprovalLevel = Literal["notify", "approve_action", "approve_plan", "take_over"]
Checkpoint = Literal["plan", "action", "final_output", "failure"]
MemoryType = Literal[
    "user_preference", "task_decision", "successful_strategy", "human_correction", "project_context"
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------------------
class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class LLMCallInfo(BaseModel):
    """Metadata about one logical LLM call (including any transport/parse retries)."""

    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    attempts: int = 1


class RunMetrics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    execution_time_s: float = 0.0
    human_review_time_s: float = 0.0
    human_escalations: int = 0
    cost_usd: float | None = None
    cost_label: str = "Cost unavailable"


# ---------------------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------------------
class UploadedFile(BaseModel):
    file_name: str
    file_type: str
    size_bytes: int
    stored_path: str


class Task(BaseModel):
    task_id: str
    trace_id: str
    user_request: str
    status: TaskStatus = "running"
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    complexity: Literal["simple", "complex"] | None = None


class SubTask(BaseModel):
    subtask_id: str
    description: str
    assigned_agent: AgentName
    dependencies: list[str] = Field(default_factory=list)
    expected_output: str = ""
    status: SubTaskStatus = "pending"
    suggested_tools: list[str] = Field(default_factory=list)
    attempts: int = 0
    reassigned_from: str | None = None
    revision_feedback: str | None = None
    human_instruction: str | None = None
    last_error: str | None = None
    parallel_group: int = 0


class Source(BaseModel):
    source_id: str = ""  # stable id derived from the URL, used for [S…] citation markers
    title: str
    url: str
    snippet: str = ""
    source: str = ""


class ToolCallRecord(BaseModel):
    call_id: str
    tool: str
    agent: str
    subtask_id: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] | None = None
    status: Literal["success", "error", "rejected", "denied", "human_provided"]
    error: str | None = None
    latency_ms: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    llm_calls: int = 0
    approved_by_human: bool | None = None
    timestamp: datetime = Field(default_factory=utcnow)


class PlannedToolCall(BaseModel):
    call_id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    approval: Literal["not_required", "pending", "approved", "modified", "rejected", "human_output"] = (
        "not_required"
    )
    human_note: str | None = None


class SubtaskToolPlan(BaseModel):
    subtask_id: str
    agent: str
    reasoning: str = ""
    confidence: float = 0.5
    calls: list[PlannedToolCall] = Field(default_factory=list)


class AgentResult(BaseModel):
    agent_name: str
    subtask_id: str
    output: str
    confidence: float = 0.0  # 0..1, agent self-assessment (one signal among several)
    tools_used: list[str] = Field(default_factory=list)
    latency: float = 0.0  # seconds
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    status: Literal["success", "failed", "human"]
    key_points: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    attempts: int = 1
    error: str | None = None


class ReviewResult(BaseModel):
    approved: bool
    score: float  # 0..10
    issues: list[str] = Field(default_factory=list)
    feedback: str
    confidence: float  # 0..1
    consistency_score: float = 0.0  # 0..10 cross-agent factual agreement
    missing_information: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    revision_targets: list[str] = Field(default_factory=list)
    reviewer_available: bool = True


class HumanDecision(BaseModel):
    decision: DecisionType
    feedback: str = ""
    modified_instruction: str = ""
    checkpoint: Checkpoint = "final_output"
    approval_level: ApprovalLevel = "approve_action"
    subject: str = ""
    review_seconds: float = 0.0
    decided_at: datetime = Field(default_factory=utcnow)


class MemoryRecord(BaseModel):
    memory_id: str
    content: str
    memory_type: MemoryType
    timestamp: datetime = Field(default_factory=utcnow)
    task_id: str = ""
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    distance: float | None = None
    embedding_dimensions: int | None = None


class ConfidenceBreakdown(BaseModel):
    review_score: float
    task_completion: float
    tool_success: float
    agent_agreement: float
    information_completeness: float
    weights: dict[str, float]
    score: float  # 0..100
    tier: Literal["auto", "notify", "approve_action", "escalate"]
    human_verified: bool = False
    notes: list[str] = Field(default_factory=list)


class FinalResult(BaseModel):
    answer: str
    confidence: float  # 0..100
    sources: list[Source] = Field(default_factory=list)
    agents_used: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    execution_time: float = 0.0
    human_verified: bool = False
    produced_by: Literal["supervisor", "human", "writer_fallback"] = "supervisor"


class ApprovalRequest(BaseModel):
    """Payload of a LangGraph interrupt; everything the Human Review panel renders."""

    request_id: str
    checkpoint: Checkpoint
    level: ApprovalLevel
    task: str
    agent: str
    proposed_action: str
    reason: str
    tool: str | None = None
    tool_input: dict[str, Any] | None = None
    confidence: float | None = None  # 0..100
    relevant_memories: list[MemoryRecord] = Field(default_factory=list)
    details: str = ""
    target_id: str | None = None
    options_help: dict[str, str] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("ev"))
    trace_id: str
    parent_id: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    node: str
    agent: str = "system"
    event_type: Literal["node", "llm_call", "tool_call", "memory", "human", "error", "retry", "info"]
    name: str
    input: str | None = None
    output: str | None = None
    tool: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    latency_ms: float = 0.0
    token_usage: TokenUsage | None = None
    llm_calls: int = 0
    model: str | None = None
    status: Literal["success", "error", "waiting", "skipped", "info"] = "success"
    error: str | None = None
    confidence: float | None = None
    human_decision: str | None = None


class RunOptions(BaseModel):
    require_plan_approval: bool = False
    require_action_approval: bool = False


# ---------------------------------------------------------------------------------------
# LLM output schemas (what Gemini is asked to return)
# ---------------------------------------------------------------------------------------
class IntakeAnalysis(BaseModel):
    complexity: Literal["simple", "complex"] = Field(
        description="simple = one specialist can answer directly; complex = needs decomposition"
    )
    reasoning: str
    memory_query: str = Field(description="Short semantic query to search long-term memory")
    user_preferences: list[str] = Field(
        description="Explicit, reusable user preferences stated in the request (format, tone, length). Empty if none."
    )
    suggested_agent: AgentName = Field(description="Best single specialist if the task is simple")


class PlannedSubtask(BaseModel):
    id: str
    description: str
    agent: str
    dependencies: list[str]
    expected_output: str
    suggested_tools: list[str]


class SupervisorPlan(BaseModel):
    reasoning: str
    subtasks: list[PlannedSubtask]
    memories_used: list[str] = Field(description="IDs of retrieved memories that influenced the plan")


class ToolCallRequest(BaseModel):
    tool: str
    arguments_json: str = Field(description="JSON object with the tool arguments")
    reason: str


class ToolPlan(BaseModel):
    reasoning: str
    confidence: float = Field(description="0-1 confidence that these tool calls are necessary and correct")
    tool_calls: list[ToolCallRequest]


class SpecialistOutput(BaseModel):
    output: str = Field(description="The deliverable for this subtask, in Markdown")
    key_points: list[str]
    confidence: float = Field(description="0-1 self-assessed confidence")
    cited_source_ids: list[str] = Field(description="IDs (e.g. S1a2b3) of provided sources actually cited")
    limitations: list[str]


class ReviewOutput(BaseModel):
    approved: bool
    score: float = Field(description="Overall quality 0-10")
    consistency_score: float = Field(description="0-10 agreement between specialist outputs")
    confidence: float = Field(description="0-1 confidence in this review")
    issues: list[str]
    missing_information: list[str]
    unsupported_claims: list[str]
    strengths: list[str]
    feedback: str = Field(description="Specific, actionable feedback for revision")
    revision_targets: list[str] = Field(description="Subtask IDs that must be redone; empty if approved")


class ReassignmentDecision(BaseModel):
    new_agent: str
    revised_description: str
    reasoning: str


class SynthesisOutput(BaseModel):
    answer: str = Field(description="Final answer in Markdown")
    summary: str = Field(description="One or two sentence summary of what was delivered")
