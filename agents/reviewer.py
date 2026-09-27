"""Reviewer Agent: structured quality gate between specialist execution and synthesis."""

from __future__ import annotations

from core.config import get_settings
from core.llm import LLMError, generate_structured, to_prompt_json
from core.schemas import AgentResult, ReviewOutput, ReviewResult, SubTask, ToolCallRecord
from core.tracing import NodeTrace

SYSTEM = """You are the Reviewer Agent, an exacting quality gate for a multi-agent system.
Evaluate the specialists' work against the user's request. Check, in order:
1. Task completion - is every requirement in the request (content, format, length, scope) satisfied?
2. Factual consistency - do the specialists' outputs agree with each other and with their tool results?
3. Unsupported claims - factual statements about the external world with no source marker [S......] and no
   supporting tool result. Label each one precisely.
4. Missing information - what a careful reader would expect but is absent.
5. Quality - structure, clarity, correctness of calculations.

Scoring: score 0-10 overall quality; consistency_score 0-10 cross-agent agreement.
approved = true only if score >= 7 AND there are no critical gaps.
You must never answer with generic praise such as "Looks good". Even when approving, list concrete strengths and
at least one residual risk or limitation in issues. feedback must be specific and actionable, naming subtask IDs.
revision_targets: IDs of subtasks whose output must be redone (empty when approved)."""


def _terminal_subtasks(subtasks: list[SubTask]) -> list[str]:
    depended_on = {d for s in subtasks for d in s.dependencies}
    return [s.subtask_id for s in subtasks if s.subtask_id not in depended_on]


def review(user_request: str, subtasks: list[SubTask], results: dict[str, AgentResult],
           tool_calls: list[ToolCallRecord], trace: NodeTrace) -> ReviewResult:
    settings = get_settings()
    work = []
    for sub in subtasks:
        result = results.get(sub.subtask_id)
        work.append({
            "subtask_id": sub.subtask_id,
            "agent": result.agent_name if result else sub.assigned_agent,
            "description": sub.description,
            "expected_output": sub.expected_output,
            "status": sub.status,
            "self_confidence": result.confidence if result else None,
            "sources_available": len(result.sources) if result else 0,
            "limitations": result.limitations if result else [],
            "output": (result.output[: settings.max_context_chars] if result else "(no output)"),
        })
    tool_summary = [
        {"tool": c.tool, "agent": c.agent, "subtask": c.subtask_id, "status": c.status, "error": c.error}
        for c in tool_calls
    ]
    prompt = (f"USER REQUEST:\n{user_request}\n\nSPECIALIST WORK:\n{to_prompt_json(work)}\n\n"
              f"TOOL CALL LOG:\n{to_prompt_json(tool_summary, 4000)}\n\n"
              f"The final deliverable is produced by subtask(s): {', '.join(_terminal_subtasks(subtasks))}.")
    valid_ids = {s.subtask_id for s in subtasks}
    try:
        output, info = generate_structured(ReviewOutput, prompt, system=SYSTEM, temperature=0.0)
    except LLMError as exc:
        trace.error("reviewer.review", exc, user_message=exc.user_message)
        return ReviewResult(
            approved=False, score=0.0, confidence=0.0, consistency_score=0.0, reviewer_available=False,
            issues=[f"Automated review could not run: {exc.user_message}"],
            feedback="The Reviewer Agent was unavailable, so this output has not been quality-checked. "
                     "Human review is required.",
        )

    score = max(0.0, min(10.0, output.score))
    targets = [t for t in output.revision_targets if t in valid_ids]
    approved = bool(output.approved and score >= 7)
    if not approved and not targets:
        targets = _terminal_subtasks(subtasks)
    feedback = output.feedback.strip()
    if len(feedback) < 40:
        feedback = (feedback + " " + "; ".join(output.issues + output.missing_information)).strip()
    result = ReviewResult(
        approved=approved, score=score, issues=output.issues, feedback=feedback,
        confidence=max(0.0, min(1.0, output.confidence)),
        consistency_score=max(0.0, min(10.0, output.consistency_score)),
        missing_information=output.missing_information, unsupported_claims=output.unsupported_claims,
        strengths=output.strengths, revision_targets=[] if approved else targets,
    )
    trace.llm("reviewer.review", info, input=f"{len(subtasks)} subtasks", output=result.model_dump(),
              agent="reviewer", confidence=result.confidence)
    return result
