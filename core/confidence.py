"""Transparent confidence scoring.

The score is not the LLM's self-reported confidence. It is a weighted blend of observable
signals, each normalized to 0..1:

  review_score              reviewer's overall score / 10
  task_completion           completed (or human-provided) subtasks / all subtasks
  tool_success              successful tool calls / attempted tool calls (neutral 1.0 if no tools ran;
                            human-denied calls are excluded, rejected/invalid requests count as failures)
  agent_agreement           0.5 * reviewer consistency score / 10
                            + 0.5 * (1 - |mean agent self-confidence - reviewer confidence|)
  information_completeness  1 - 0.2 * missing items - 0.1 * unsupported claims  (floored at 0)

score = 100 * sum(weight_i * signal_i) / sum(weights). Weights and tier thresholds come from config.
Human feedback is reported separately (`human_verified`) and never inflates the computed score.
"""

from __future__ import annotations

from core.config import get_settings
from core.schemas import AgentResult, ConfidenceBreakdown, HumanDecision, ReviewResult, SubTask, ToolCallRecord


def tier_for(score: float) -> str:
    settings = get_settings()
    if score >= settings.auto_threshold:
        return "auto"
    if score >= settings.notify_threshold:
        return "notify"
    if score >= settings.approve_threshold:
        return "approve_action"
    return "escalate"


def compute_confidence(review: ReviewResult | None, subtasks: list[SubTask], results: dict[str, AgentResult],
                       tool_calls: list[ToolCallRecord], human_decisions: list[HumanDecision]) -> ConfidenceBreakdown:
    settings = get_settings()
    notes: list[str] = []

    review_score = (review.score / 10) if review else 0.0
    if review and not review.reviewer_available:
        notes.append("Reviewer unavailable: review score is 0.")

    total = len(subtasks) or 1
    done = sum(1 for s in subtasks if s.status == "completed")
    task_completion = done / total
    if done < len(subtasks):
        notes.append(f"{len(subtasks) - done} of {len(subtasks)} subtasks not completed.")

    attempted = [c for c in tool_calls if c.status in ("success", "error", "rejected", "human_provided")]
    if attempted:
        ok = sum(1 for c in attempted if c.status in ("success", "human_provided"))
        tool_success = ok / len(attempted)
        if ok < len(attempted):
            notes.append(f"{len(attempted) - ok} of {len(attempted)} tool calls failed or were invalid.")
    else:
        tool_success = 1.0
        notes.append("No tools executed; tool success treated as neutral (1.0).")

    agent_confs = [r.confidence for r in results.values() if r.status == "success"]
    if review and agent_confs:
        mean_agent = sum(agent_confs) / len(agent_confs)
        agent_agreement = 0.5 * (review.consistency_score / 10) + 0.5 * (1 - abs(mean_agent - review.confidence))
    elif review:
        agent_agreement = review.consistency_score / 10
    else:
        agent_agreement = 0.0

    if review:
        information_completeness = max(
            0.0, 1 - 0.2 * len(review.missing_information) - 0.1 * len(review.unsupported_claims)
        )
    else:
        information_completeness = 0.0

    weights = settings.confidence_weights.as_dict()
    signals = {
        "review_score": review_score, "task_completion": task_completion, "tool_success": tool_success,
        "agent_agreement": agent_agreement, "information_completeness": information_completeness,
    }
    weight_sum = sum(weights.values()) or 1.0
    score = round(100 * sum(weights[k] * signals[k] for k in weights) / weight_sum, 1)

    human_verified = any(d.checkpoint == "final_output" and d.decision in ("approve", "take_over")
                         for d in human_decisions)
    if any(d.decision in ("modify", "take_over") for d in human_decisions):
        notes.append("Human feedback was applied during this run.")

    return ConfidenceBreakdown(
        **{k: round(v, 3) for k, v in signals.items()}, weights=weights, score=score,
        tier=tier_for(score), human_verified=human_verified, notes=notes,  # type: ignore[arg-type]
    )
