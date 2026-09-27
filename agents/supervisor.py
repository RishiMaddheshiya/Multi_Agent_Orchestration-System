"""Supervisor Agent: intake analysis, planning, failure reassignment and final synthesis.

The Supervisor orchestrates but never executes subtasks. It only assigns registered
specialists and suggests registered tools. graph.workflow.plan_validation enforces this
again deterministically, so the rule doesn't depend on the LLM following instructions.
"""

from __future__ import annotations

import re

from agents.analyst import AnalysisAgent
from agents.base import SpecialistAgent
from agents.researcher import ResearchAgent
from agents.writer import WriterAgent
from core.config import get_settings
from core.llm import generate_structured, to_prompt_json
from core.schemas import (
    AgentResult,
    IntakeAnalysis,
    LLMCallInfo,
    MemoryRecord,
    ReassignmentDecision,
    ReviewResult,
    Source,
    SubTask,
    SupervisorPlan,
    SynthesisOutput,
    UploadedFile,
)
from tools.registry import REGISTRY

SPECIALIST_AGENTS: dict[str, SpecialistAgent] = {
    agent.name: agent for agent in (ResearchAgent(), AnalysisAgent(), WriterAgent())
}


def team_description() -> str:
    lines = []
    for agent in SPECIALIST_AGENTS.values():
        tools = ", ".join(agent.allowed_tools()) or "none"
        lines.append(f"- {agent.name} ({agent.label}): {agent.role} Tools: {tools}.")
    return "\n".join(lines)


INTAKE_SYSTEM = """You are the Supervisor of a multi-agent system. Classify the user's request.
simple: a single specialist can fully answer it (a factual lookup, one calculation, a short rewrite).
complex: it needs several distinct steps (research + analysis + writing, multi-part deliverables).
memory_query: a short semantic query capturing the topic and deliverable type, for searching past work.
user_preferences: only explicit, reusable preferences (e.g. "use bullet points", "keep under 300 words").
Team:
{team}"""

PLAN_SYSTEM = """You are the Supervisor of a multi-agent system. Decompose the user's request into the minimum set
of subtasks (at most {max_subtasks}) and assign each to exactly one specialist from the team below.

Team:
{team}

Registered tools (suggested_tools must come ONLY from this list and must be allowed for the assigned agent):
{tools}

Rules:
- ids are short strings like "t1", "t2".
- dependencies list the ids whose outputs a subtask needs. Subtasks without dependencies run in parallel, so
  only add a dependency when the output is truly needed.
- Research before analysis before writing when all three are needed. The final deliverable must come from a
  single writer subtask that depends on the work it summarizes.
- expected_output states concretely what the subtask must return.
- Retrieved memories are hints: use those that apply (list their ids in memories_used) and ignore the rest.
- If a human gave plan feedback, follow it exactly."""

REASSIGN_SYSTEM = """You are the Supervisor. A subtask failed after retries. Choose the specialist best able to
complete it differently (it may be a different agent than before) and rewrite the description so it is achievable
with that agent's tools. Team:
{team}"""

SYNTHESIS_SYSTEM = """You are the Supervisor producing the final answer for the user from the team's approved work.
Rules:
- Use only information in the specialist outputs. Do not add facts.
- Follow the user's requested format and preferences. If a writer deliverable exists, build on it rather than
  rewriting from scratch; integrate any reviewer caveats that affect correctness.
- Keep source markers like [S1a2b3] exactly as written next to the facts they support; never write raw URLs.
- If important parts of the request could not be completed, say so briefly at the end under "Limitations"."""


def analyze_request(user_request: str, uploads: list[UploadedFile]) -> tuple[IntakeAnalysis, LLMCallInfo]:
    prompt = f"USER REQUEST:\n{user_request}"
    if uploads:
        prompt += "\n\nUPLOADED FILES: " + ", ".join(u.file_name for u in uploads)
    return generate_structured(IntakeAnalysis, prompt, system=INTAKE_SYSTEM.format(team=team_description()),
                               temperature=0.0)


def create_plan(user_request: str, intake: IntakeAnalysis | None, memories: list[MemoryRecord],
                uploads: list[UploadedFile], plan_feedback: str | None,
                previous_plan: list[SubTask] | None) -> tuple[SupervisorPlan, LLMCallInfo]:
    settings = get_settings()
    parts = [f"USER REQUEST:\n{user_request}"]
    if intake and intake.user_preferences:
        parts.append("USER PREFERENCES:\n- " + "\n- ".join(intake.user_preferences))
    if uploads:
        parts.append("UPLOADED FILES:\n" + "\n".join(f"- {u.file_name} ({u.file_type})" for u in uploads))
    if memories:
        parts.append("RETRIEVED LONG-TERM MEMORIES:\n" + "\n".join(
            f"- id={m.memory_id} type={m.memory_type} distance={m.distance}: {m.content}" for m in memories))
    if previous_plan:
        parts.append("PREVIOUS PLAN:\n" + to_prompt_json([
            {"id": s.subtask_id, "agent": s.assigned_agent, "description": s.description,
             "dependencies": s.dependencies} for s in previous_plan]))
    if plan_feedback:
        parts.append(f"HUMAN PLAN FEEDBACK (must be applied):\n{plan_feedback}")
    system = PLAN_SYSTEM.format(max_subtasks=settings.max_subtasks, team=team_description(),
                                tools=REGISTRY.catalog())
    return generate_structured(SupervisorPlan, "\n\n".join(parts), system=system, temperature=0.1)


def reassign(user_request: str, subtask: SubTask, error: str) -> tuple[ReassignmentDecision, LLMCallInfo]:
    prompt = (f"USER REQUEST:\n{user_request}\n\nFAILED SUBTASK {subtask.subtask_id} "
              f"(agent: {subtask.assigned_agent}):\n{subtask.description}\nExpected: {subtask.expected_output}\n\n"
              f"ERROR:\n{error}")
    return generate_structured(ReassignmentDecision, prompt, system=REASSIGN_SYSTEM.format(team=team_description()),
                               temperature=0.2)


def synthesize(user_request: str, preferences: list[str], subtasks: list[SubTask],
               results: dict[str, AgentResult], review: ReviewResult | None,
               human_notes: list[str]) -> tuple[SynthesisOutput, LLMCallInfo]:
    settings = get_settings()
    work = []
    for sub in subtasks:
        result = results.get(sub.subtask_id)
        if result and result.status != "failed" and result.output:
            work.append(f"### {sub.subtask_id} - {result.agent_name}: {sub.description}\n"
                        f"{result.output[: settings.max_context_chars * 2]}")
    parts = [f"USER REQUEST:\n{user_request}", "SPECIALIST OUTPUTS:\n" + "\n\n".join(work)]
    if preferences:
        parts.append("USER PREFERENCES:\n- " + "\n- ".join(preferences))
    if review:
        parts.append(f"REVIEWER ASSESSMENT (score {review.score}/10): {review.feedback}\n"
                     f"Unsupported claims flagged: {review.unsupported_claims or 'none'}")
    if human_notes:
        parts.append("HUMAN NOTES:\n- " + "\n- ".join(human_notes))
    return generate_structured(SynthesisOutput, "\n\n".join(parts), system=SYNTHESIS_SYSTEM, temperature=0.2)


MARKER_RE = re.compile(r"\[(S[0-9a-f]{6})\]")


def renumber_citations(answer: str, sources: list[Source]) -> tuple[str, list[Source]]:
    """Turn [S1a2b3] markers into [1], [2] ... in order of first use; drop unknown markers."""
    by_id = {s.source_id: s for s in sources}
    order: list[str] = []

    def repl(match: re.Match) -> str:
        sid = match.group(1)
        if sid not in by_id:
            return ""
        if sid not in order:
            order.append(sid)
        return f"[{order.index(sid) + 1}]"

    text = MARKER_RE.sub(repl, answer)
    return text, [by_id[sid] for sid in order]
