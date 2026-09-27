"""LangGraph state machine for the orchestration system, plus the runner the UI drives.

    START
      -> task_intake -> memory_retrieval -> supervisor -> plan_validation -> plan_approval
      plan_approval: approved -> specialist_dispatch | modify -> supervisor | reject -> finalize
                     take over -> synthesis
      specialist_dispatch: ready wave -> tool_planning (parallel Send per subtask) | all done -> reviewer
      tool_planning -> action_approval (loops once per sensitive call awaiting approval)
                    -> specialist_execution (parallel Send per subtask) -> result_collection
      result_collection: failures -> reassignment (retry via dispatch) or failure_escalation (human)
                         otherwise -> specialist_dispatch (next wave)
      reviewer -> confidence_check
      confidence_check: rejected & revisions left -> revise -> specialist_dispatch
                        auto / notify -> synthesis | approve_action / escalate -> human_review
      human_review: approve -> synthesis | modify -> specialist_dispatch | reject -> finalize
                    take over -> synthesis
      synthesis -> finalize -> END

Human-in-the-loop uses `interrupt()` with a checkpointer: the graph really stops and later
resumes with `Command(resume=decision)`. Each gate node handles one decision per execution
and loops back to itself, and all code before an interrupt is deterministic, so the node is
safe to re-execute when it resumes.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Overwrite, Send, interrupt
from pydantic import BaseModel

from agents import supervisor as sup
from agents.base import AgentContext
from agents.reviewer import review as run_review
from agents.supervisor import SPECIALIST_AGENTS
from core import schemas
from core.config import get_logger, get_settings
from core.confidence import compute_confidence
from core.llm import LLMConfigError, LLMError
from core.schemas import (
    AGENT_LABELS,
    SPECIALISTS,
    AgentResult,
    ApprovalRequest,
    FinalResult,
    HumanDecision,
    IntakeAnalysis,
    MemoryRecord,
    PlannedSubtask,
    RunOptions,
    SubTask,
    SubtaskToolPlan,
    SupervisorPlan,
    Task,
    ToolCallRecord,
    UploadedFile,
    new_id,
    utcnow,
)
from core.tracing import NodeTrace, compute_metrics
from memory import short_term
from memory.short_term import AgentState, RunSnapshot, snapshot_from_state
from tools.registry import REGISTRY, ToolContext

log = get_logger("workflow")

AGENT_ALIASES = {
    "research": "researcher", "research_agent": "researcher", "researcher_agent": "researcher",
    "analysis": "analyst", "analyzer": "analyst", "analysis_agent": "analyst", "analyst_agent": "analyst",
    "writing": "writer", "writer_agent": "writer", "writing_agent": "writer",
}
TERMINAL = {"completed", "failed", "skipped"}


# =========================================================================================
# helpers
# =========================================================================================
def _subtasks(state: AgentState) -> list[SubTask]:
    return list((state.get("subtasks") or {}).values())


def _trace_id(state: AgentState) -> str:
    return state["task"].trace_id


def _agent_context(state: AgentState, subtask: SubTask) -> AgentContext:
    subs = state.get("subtasks") or {}
    results = state.get("results") or {}
    intake = state.get("intake")
    used = set(state.get("memories_used") or [])
    memories = [m for m in state.get("retrieved_memories") or [] if not used or m.memory_id in used]
    return AgentContext(
        trace_id=_trace_id(state), task_id=state["task"].task_id, user_request=state["task"].user_request,
        user_preferences=intake.user_preferences if intake else [],
        dependency_results=[(subs[d], results.get(d)) for d in subtask.dependencies if d in subs],
        memories=memories, uploads=state.get("uploads") or [],
    )


def _dependents(subtasks: list[SubTask], roots: set[str]) -> set[str]:
    found = set(roots)
    changed = True
    while changed:
        changed = False
        for s in subtasks:
            if s.subtask_id not in found and found.intersection(s.dependencies):
                found.add(s.subtask_id)
                changed = True
    return found


def _terminal_ids(subtasks: list[SubTask]) -> list[str]:
    depended = {d for s in subtasks for d in s.dependencies}
    return [s.subtask_id for s in subtasks if s.subtask_id not in depended]


def _decision(value: Any, *, checkpoint: str, level: str, subject: str) -> HumanDecision:
    value = value if isinstance(value, dict) else {"decision": str(value)}
    decision = HumanDecision(
        decision=value.get("decision", "approve"), feedback=value.get("feedback", "") or "",
        modified_instruction=(value.get("modified_instruction", "") or "").strip(),
        checkpoint=checkpoint, approval_level=level, subject=subject[:300],  # type: ignore[arg-type]
        review_seconds=float(value.get("review_seconds", 0) or 0),
    )
    if decision.decision in ("modify", "take_over") and not decision.modified_instruction:
        # A modify/take-over without content cannot be applied; treat it as feedback-only approval.
        decision.feedback = (decision.feedback + " (empty instruction; treated as approve)").strip()
        decision.decision = "approve"
    return decision


def _plan_markdown(subtasks: list[SubTask]) -> str:
    rows = ["| ID | Agent | Depends on | Group | Description |", "|---|---|---|---|---|"]
    for s in subtasks:
        rows.append(f"| {s.subtask_id} | {s.assigned_agent} | {', '.join(s.dependencies) or '-'} | "
                    f"{s.parallel_group} | {s.description.replace('|', '/')} |")
    return "\n".join(rows)


# =========================================================================================
# nodes
# =========================================================================================
def task_intake(state: AgentState) -> dict:
    task = state["task"]
    t = NodeTrace(task.trace_id, "task_intake", "supervisor", input=task.user_request)
    notes: list[str] = []
    try:
        intake, info = sup.analyze_request(task.user_request, state.get("uploads") or [])
        t.llm("supervisor.analyze_request", info, input=task.user_request, output=intake.model_dump())
        route = "continue"
    except LLMConfigError as exc:
        t.error("supervisor.analyze_request", exc, user_message=exc.user_message)
        return {"route": "abort", "notifications": [exc.user_message],
                "trace": t.events(status="error", error=exc.user_message)}
    except LLMError as exc:
        t.error("supervisor.analyze_request", exc, user_message=exc.user_message)
        intake = IntakeAnalysis(complexity="complex", reasoning="Fallback: intake classification unavailable.",
                                memory_query=task.user_request[:300], user_preferences=[], suggested_agent="writer")
        notes.append("Task classification failed; treating the task as complex.")
        route = "continue"
    task = task.model_copy(update={"complexity": intake.complexity})
    return {"task": task, "intake": intake, "route": route, "notifications": notes,
            "trace": t.events(output=f"{intake.complexity}: {intake.reasoning}")}


def memory_retrieval(state: AgentState) -> dict:
    settings = get_settings()
    intake = state.get("intake")
    query = intake.memory_query if intake else state["task"].user_request
    t = NodeTrace(_trace_id(state), "memory_retrieval", "supervisor", input=query)
    if not settings.memory_enabled:
        t.info("memory.disabled", "Long-term memory disabled by configuration.")
        return {"retrieved_memories": [], "trace": t.events(status="skipped")}
    started = time.perf_counter()
    try:
        from memory.long_term import get_long_term_memory

        memories = get_long_term_memory().search(query)
        t.memory("long_term.search", input=query,
                 output=[{"id": m.memory_id, "type": m.memory_type, "distance": m.distance} for m in memories],
                 latency_ms=(time.perf_counter() - started) * 1000)
        return {"retrieved_memories": memories, "trace": t.events(output=f"{len(memories)} memories retrieved")}
    except Exception as exc:  # noqa: BLE001 - memory is an enhancement, never a hard dependency
        message = getattr(exc, "user_message", None) or str(exc)
        t.memory("long_term.search", input=query, status="error", error=message,
                 latency_ms=(time.perf_counter() - started) * 1000)
        return {"retrieved_memories": [], "trace": t.events(status="error", error=message),
                "notifications": ["Long-term memory was unavailable; continuing without it."]}


def supervisor_plan(state: AgentState) -> dict:
    task = state["task"]
    intake = state.get("intake")
    feedback = state.get("plan_feedback")
    t = NodeTrace(task.trace_id, "supervisor", "supervisor", input=feedback or task.user_request)
    notes: list[str] = []
    if intake and intake.complexity == "simple" and not feedback:
        raw = SupervisorPlan(
            reasoning=f"Simple task: routed directly to the {intake.suggested_agent} without decomposition.",
            subtasks=[PlannedSubtask(id="t1", description=task.user_request, agent=intake.suggested_agent,
                                     dependencies=[], expected_output="A complete, direct answer to the request.",
                                     suggested_tools=[])],
            memories_used=[],
        )
        t.info("supervisor.route_simple", raw.reasoning)
    else:
        previous = _subtasks(state) if feedback else None
        try:
            raw, info = sup.create_plan(task.user_request, intake, state.get("retrieved_memories") or [],
                                        state.get("uploads") or [], feedback, previous)
            t.llm("supervisor.create_plan", info, input=task.user_request, output=raw.model_dump())
        except LLMError as exc:
            t.error("supervisor.create_plan", exc, user_message=exc.user_message)
            notes.append("The Supervisor could not generate a plan; using the standard "
                         "research -> analysis -> writing template.")
            raw = SupervisorPlan(
                reasoning="Fallback template plan (Supervisor planning call failed).",
                subtasks=[
                    PlannedSubtask(id="t1", agent="researcher", dependencies=[], suggested_tools=["web_search"],
                                   description=f"Gather the facts needed for: {task.user_request}",
                                   expected_output="Sourced research notes"),
                    PlannedSubtask(id="t2", agent="analyst", dependencies=["t1"], suggested_tools=["calculator"],
                                   description="Analyze the research: patterns, comparisons, key numbers",
                                   expected_output="Structured analysis"),
                    PlannedSubtask(id="t3", agent="writer", dependencies=["t1", "t2"], suggested_tools=[],
                                   description="Write the deliverable the user requested",
                                   expected_output="Final deliverable in the requested format"),
                ],
                memories_used=[],
            )
    return {"raw_plan": raw, "plan_reasoning": raw.reasoning, "notifications": notes,
            "trace": t.events(output=f"{len(raw.subtasks)} subtasks proposed")}


def plan_validation(state: AgentState) -> dict:
    """Deterministic guardrails over the Supervisor's plan."""
    settings = get_settings()
    raw: SupervisorPlan = state["raw_plan"]  # type: ignore[typeddict-item]
    t = NodeTrace(_trace_id(state), "plan_validation", "supervisor", input=f"{len(raw.subtasks)} proposed subtasks")
    issues: list[str] = []
    if len(raw.subtasks) > settings.max_subtasks:
        issues.append(f"Plan truncated from {len(raw.subtasks)} to {settings.max_subtasks} subtasks.")

    id_map: dict[str, str] = {}
    staged: list[tuple[str, PlannedSubtask, str]] = []
    for idx, planned in enumerate(raw.subtasks[: settings.max_subtasks], start=1):
        sid = re.sub(r"[^A-Za-z0-9_-]", "", planned.id or "") or f"t{idx}"
        if sid in id_map.values():
            sid = f"t{idx}"
            issues.append(f"Duplicate subtask id '{planned.id}' renamed to {sid}.")
        id_map[planned.id] = sid
        agent = planned.agent.strip().lower().replace(" ", "_")
        agent = AGENT_ALIASES.get(agent, agent)
        if agent not in SPECIALISTS:
            issues.append(f"{sid}: unknown agent '{planned.agent}' replaced with writer.")
            agent = "writer"
        staged.append((sid, planned, agent))

    subtasks: dict[str, SubTask] = {}
    for sid, planned, agent in staged:
        allowed = {spec.name for spec in REGISTRY.for_agent(agent)}
        tools = []
        for tool in planned.suggested_tools:
            if REGISTRY.get(tool) is None:
                issues.append(f"{sid}: unregistered tool '{tool}' removed.")
            elif tool not in allowed:
                issues.append(f"{sid}: tool '{tool}' not permitted for {agent}; removed.")
            else:
                tools.append(tool)
        deps = []
        for dep in planned.dependencies:
            mapped = id_map.get(dep, dep)
            if mapped == sid:
                issues.append(f"{sid}: self-dependency removed.")
            elif mapped not in {s for s, _, _ in staged}:
                issues.append(f"{sid}: unknown dependency '{dep}' removed.")
            elif mapped not in deps:
                deps.append(mapped)
        subtasks[sid] = SubTask(subtask_id=sid, description=planned.description.strip() or "Complete this step",
                                assigned_agent=agent, dependencies=deps,  # type: ignore[arg-type]
                                expected_output=planned.expected_output, suggested_tools=tools)

    # Break dependency cycles (drop the back edge) and compute parallel groups.
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(sid: str) -> None:
        visiting.add(sid)
        for dep in list(subtasks[sid].dependencies):
            if dep in visiting:
                subtasks[sid].dependencies.remove(dep)
                issues.append(f"Dependency cycle broken: {sid} no longer waits for {dep}.")
            elif dep not in done:
                visit(dep)
        visiting.discard(sid)
        done.add(sid)

    for sid in list(subtasks):
        if sid not in done:
            visit(sid)
    levels: dict[str, int] = {}

    def level(sid: str) -> int:
        if sid not in levels:
            deps = subtasks[sid].dependencies
            levels[sid] = 0 if not deps else 1 + max(level(d) for d in deps)
        return levels[sid]

    for sid, sub in subtasks.items():
        sub.parallel_group = level(sid)

    if not subtasks:
        issues.append("Plan was empty; a single writer subtask was created.")
        subtasks["t1"] = SubTask(subtask_id="t1", description=state["task"].user_request, assigned_agent="writer",
                                 expected_output="A complete answer to the request.")

    retrieved = {m.memory_id for m in state.get("retrieved_memories") or []}
    used = [m for m in raw.memories_used if m in retrieved]
    for issue in issues:
        t.info("plan_validation.issue", issue, status="error")
    groups = max((s.parallel_group for s in subtasks.values()), default=0) + 1
    return {
        "subtasks": Overwrite(subtasks), "plan_history": [list(subtasks.values())], "plan_issues": issues,
        "plan_version": (state.get("plan_version") or 0) + 1, "plan_status": "pending",
        "memories_used": used, "results": Overwrite({}), "tool_plans": Overwrite({}), "wave": [],
        "escalations": [], "revision_count": 0, "review": None,
        "notifications": [f"Plan validation adjusted the plan: {i}" for i in issues],
        "trace": t.events(output=f"{len(subtasks)} subtasks in {groups} execution group(s); {len(issues)} issue(s)"),
    }


def plan_approval(state: AgentState) -> dict:
    options = state.get("options") or RunOptions()
    if not options.require_plan_approval:
        return {"plan_status": "approved"}
    subtasks = _subtasks(state)
    version = state.get("plan_version") or 1
    used = set(state.get("memories_used") or [])
    request = ApprovalRequest(
        request_id=f"plan-v{version}", checkpoint="plan", level="approve_plan", task=state["task"].user_request,
        agent="supervisor", proposed_action=f"Execute a {len(subtasks)}-subtask plan (version {version})",
        reason="Plan approval is enabled for this run. " + (state.get("plan_reasoning") or ""),
        relevant_memories=[m for m in state.get("retrieved_memories") or [] if m.memory_id in used],
        details=_plan_markdown(subtasks),
        options_help={"approve": "Approve Plan: execute as shown", "modify": "Modify Plan: the Supervisor re-plans "
                      "with your instructions", "reject": "Reject Plan: stop the task",
                      "take_over": "Take Over: write the final answer yourself"},
    )
    value = interrupt(request.model_dump(mode="json"))
    decision = _decision(value, checkpoint="plan", level="approve_plan", subject=request.proposed_action)
    t = NodeTrace(_trace_id(state), "plan_approval", "human")
    t.human(decision)
    update: dict[str, Any] = {"human_decisions": [decision], "trace": t.events(output=decision.decision)}
    if decision.decision == "approve":
        update["plan_status"] = "approved"
    elif decision.decision == "modify":
        update.update(plan_status="modify", plan_feedback=decision.modified_instruction,
                      human_notes=[f"Plan feedback: {decision.modified_instruction}"])
    elif decision.decision == "reject":
        update.update(plan_status="rejected", route="abort")
    else:
        update.update(plan_status="taken_over", human_final_override=decision.modified_instruction)
    return update


def specialist_dispatch(state: AgentState) -> dict:
    subtasks = _subtasks(state)
    status = {s.subtask_id: s.status for s in subtasks}
    t = NodeTrace(_trace_id(state), "specialist_dispatch", "supervisor")
    ready = [s for s in subtasks if s.status == "pending" and all(status.get(d) in TERMINAL for d in s.dependencies)]
    updates = {s.subtask_id: s.model_copy(update={"status": "running"}) for s in ready}
    if not ready:
        blocked = [s for s in subtasks if s.status == "pending"]
        for s in blocked:  # unreachable after validation; guard against deadlock anyway
            updates[s.subtask_id] = s.model_copy(update={"status": "skipped", "last_error": "Blocked dependencies"})
        message = "All subtasks finished; sending to Reviewer." if not blocked else "Blocked subtasks skipped."
    else:
        message = (f"Dispatching {', '.join(s.subtask_id for s in ready)}"
                   + (" in parallel" if len(ready) > 1 else ""))
    t.info("dispatch", message)
    return {"subtasks": updates, "wave": [s.subtask_id for s in ready], "trace": t.events(output=message)}


def tool_planning(state: AgentState) -> dict:
    sid = state["current_subtask_id"]
    subtask = state["subtasks"][sid]
    agent = SPECIALIST_AGENTS[subtask.assigned_agent]
    options = state.get("options") or RunOptions()
    t = NodeTrace(_trace_id(state), "tool_planning", agent.name, input=subtask.description)
    try:
        plan, rejected = agent.plan_tools(subtask, _agent_context(state, subtask), t, options.require_action_approval)
    except LLMError as exc:
        t.error(f"{agent.name}.plan_tools", exc, user_message=exc.user_message)
        plan = SubtaskToolPlan(subtask_id=sid, agent=agent.name,
                               reasoning=f"Tool planning failed ({exc.user_message}); continuing without tools.")
        rejected = []
    summary = ", ".join(f"{c.tool}{'*' if c.approval == 'pending' else ''}" for c in plan.calls) or "no tools"
    return {"tool_plans": {sid: plan}, "tool_calls": rejected,
            "trace": t.events(output=f"{sid}: {summary}", confidence=plan.confidence)}


def _pending_action(state: AgentState) -> tuple[str, int] | None:
    plans = state.get("tool_plans") or {}
    for sid in state.get("wave") or []:
        plan = plans.get(sid)
        if plan:
            for idx, call in enumerate(plan.calls):
                if call.approval == "pending":
                    return sid, idx
    return None


def action_approval(state: AgentState) -> dict:
    pending = _pending_action(state)
    if pending is None:
        return {"route": "execute"}
    sid, idx = pending
    plan = state["tool_plans"][sid]
    call = plan.calls[idx]
    subtask = state["subtasks"][sid]
    spec = REGISTRY.get(call.tool)
    used = set(state.get("memories_used") or [])
    request = ApprovalRequest(
        request_id=call.call_id, checkpoint="action", level="approve_action", task=state["task"].user_request,
        agent=plan.agent, proposed_action=f"Call `{call.tool}` for subtask {sid}", reason=call.reason or plan.reasoning,
        tool=call.tool, tool_input=call.arguments, confidence=round(plan.confidence * 100, 1),
        relevant_memories=[m for m in state.get("retrieved_memories") or [] if m.memory_id in used],
        details=f"Subtask {sid}: {subtask.description}", target_id=call.call_id,
        options_help={"approve": "Run the tool call as proposed",
                      "modify": f"Replace '{spec.primary_arg if spec else 'arguments'}' (or paste a JSON object)",
                      "reject": "Skip this call; the agent continues without it",
                      "take_over": "Provide the tool result yourself"},
    )
    value = interrupt(request.model_dump(mode="json"))
    decision = _decision(value, checkpoint="action", level="approve_action", subject=request.proposed_action)
    t = NodeTrace(_trace_id(state), "action_approval", "human")
    t.human(decision)
    notes: list[str] = []
    new_call = call.model_copy()
    if decision.decision == "approve":
        new_call.approval = "approved"
    elif decision.decision == "reject":
        new_call.approval, new_call.human_note = "rejected", decision.feedback or "Rejected by reviewer"
    elif decision.decision == "take_over":
        new_call.approval, new_call.human_note = "human_output", decision.modified_instruction
    else:
        try:
            args = json.loads(decision.modified_instruction)
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            args = {**call.arguments, (spec.primary_arg if spec else "query"): decision.modified_instruction}
        validated, problem = REGISTRY.validate_call(plan.agent, call.tool, args)
        if problem:
            notes.append(f"Modified arguments were invalid ({problem}); please review the action again.")
        else:
            new_call.arguments = validated.model_dump(mode="json")
            new_call.approval, new_call.human_note = "modified", decision.modified_instruction
    calls = list(plan.calls)
    calls[idx] = new_call
    return {"tool_plans": {sid: plan.model_copy(update={"calls": calls})}, "human_decisions": [decision],
            "notifications": notes, "route": "loop", "trace": t.events(output=f"{call.tool}: {decision.decision}")}


def specialist_execution(state: AgentState) -> dict:
    sid = state["current_subtask_id"]
    subtask = state["subtasks"][sid]
    agent = SPECIALIST_AGENTS[subtask.assigned_agent]
    plan = (state.get("tool_plans") or {}).get(sid)
    t = NodeTrace(_trace_id(state), "specialist_execution", agent.name, input=subtask.description)
    context = ToolContext(task_id=state["task"].task_id, uploads=state.get("uploads") or [])
    records: list[ToolCallRecord] = []
    for call in plan.calls if plan else []:
        if call.approval == "rejected":
            record = ToolCallRecord(call_id=call.call_id, tool=call.tool, agent=agent.name, subtask_id=sid,
                                    input=call.arguments, status="denied", approved_by_human=False,
                                    error=f"Denied by human: {call.human_note}")
        elif call.approval == "human_output":
            record = ToolCallRecord(call_id=call.call_id, tool=call.tool, agent=agent.name, subtask_id=sid,
                                    input=call.arguments, status="human_provided", approved_by_human=True,
                                    output={"human_provided_result": call.human_note})
        else:
            record = REGISTRY.execute(
                call_id=call.call_id, agent=agent.name, subtask_id=sid, tool=call.tool, arguments=call.arguments,
                context=context, approved_by_human=True if call.approval in ("approved", "modified") else None,
            )
        t.tool(record)
        records.append(record)

    result = agent.execute(subtask, _agent_context(state, subtask), records, t)
    ok = result.status == "success"
    updated = subtask.model_copy(update={"status": "completed" if ok else "failed",
                                         "attempts": subtask.attempts + result.attempts,
                                         "last_error": None if ok else result.error})
    return {"results": {sid: result}, "subtasks": {sid: updated}, "tool_calls": records,
            "trace": t.events(status="success" if ok else "error", output=result.output[:600] or result.error,
                              error=result.error, confidence=result.confidence)}


def result_collection(state: AgentState) -> dict:
    subtasks = state["subtasks"]
    t = NodeTrace(_trace_id(state), "result_collection", "supervisor")
    updates: dict[str, SubTask] = {}
    escalations: list[str] = []
    notes: list[str] = []
    fallback = {"researcher": "analyst", "analyst": "researcher", "writer": "analyst"}
    for sid in state.get("wave") or []:
        sub = subtasks[sid]
        if sub.status != "failed":
            continue
        label = AGENT_LABELS.get(sub.assigned_agent, sub.assigned_agent)
        if sub.reassigned_from is None:
            try:
                choice, info = sup.reassign(state["task"].user_request, sub, sub.last_error or "unknown error")
                t.llm("supervisor.reassign", info, input=sid, output=choice.model_dump())
                agent = AGENT_ALIASES.get(choice.new_agent.strip().lower(), choice.new_agent.strip().lower())
                if agent not in SPECIALISTS:
                    agent = fallback[sub.assigned_agent]
                description = choice.revised_description.strip() or sub.description
            except LLMError as exc:
                t.error("supervisor.reassign", exc, user_message=exc.user_message)
                agent, description = fallback[sub.assigned_agent], sub.description
            updates[sid] = sub.model_copy(update={"assigned_agent": agent, "description": description,
                                                  "status": "pending", "reassigned_from": sub.assigned_agent})
            notes.append(f"The {label} failed on {sid}. The Supervisor will retry using an alternative strategy "
                         f"({AGENT_LABELS.get(agent, agent)}).")
            t.info("reassign", f"{sid}: {sub.assigned_agent} -> {agent}", event_type="retry", status="error",
                   error=sub.last_error)
        else:
            escalations.append(sid)
            notes.append(f"{sid} failed after retries and reassignment. Human input is required.")
    done = sum(1 for s in subtasks.values() if s.status == "completed")
    return {"subtasks": updates, "escalations": escalations, "wave": [], "notifications": notes,
            "trace": t.events(output=f"{done}/{len(subtasks)} subtasks completed; {len(escalations)} escalated")}


def failure_escalation(state: AgentState) -> dict:
    pending = list(state.get("escalations") or [])
    if not pending:
        return {"route": "dispatch"}
    sid = pending[0]
    sub = state["subtasks"][sid]
    request = ApprovalRequest(
        request_id=f"failure-{sid}-{sub.attempts}", checkpoint="failure", level="take_over",
        task=state["task"].user_request, agent=sub.assigned_agent,
        proposed_action=f"Subtask {sid} could not be completed automatically",
        reason=f"Failed after {sub.attempts} attempt(s) and reassignment from {sub.reassigned_from}. "
               f"Last error: {sub.last_error}",
        confidence=0.0, details=f"{sub.description}\n\nExpected output: {sub.expected_output}", target_id=sid,
        options_help={"approve": "Skip this subtask and continue without it",
                      "modify": "Retry the subtask with your instructions",
                      "reject": "Abort the whole task", "take_over": "Provide this subtask's output yourself"},
    )
    value = interrupt(request.model_dump(mode="json"))
    decision = _decision(value, checkpoint="failure", level="take_over", subject=request.proposed_action)
    t = NodeTrace(_trace_id(state), "failure_escalation", "human")
    t.human(decision)
    update: dict[str, Any] = {"escalations": pending[1:], "human_decisions": [decision], "route": "continue",
                              "trace": t.events(output=f"{sid}: {decision.decision}")}
    if decision.decision == "approve":
        update["subtasks"] = {sid: sub.model_copy(update={"status": "skipped"})}
    elif decision.decision == "modify":
        update["subtasks"] = {sid: sub.model_copy(update={"status": "pending",
                                                          "human_instruction": decision.modified_instruction})}
    elif decision.decision == "reject":
        update["route"] = "abort"
    else:
        update["subtasks"] = {sid: sub.model_copy(update={"status": "completed"})}
        update["results"] = {sid: AgentResult(agent_name="human", subtask_id=sid, status="human", confidence=1.0,
                                              output=decision.modified_instruction)}
    return update


def reviewer(state: AgentState) -> dict:
    t = NodeTrace(_trace_id(state), "reviewer", "reviewer")
    review = run_review(state["task"].user_request, _subtasks(state), state.get("results") or {},
                        state.get("tool_calls") or [], t)
    return {"review": review, "review_history": [review],
            "trace": t.events(status="success" if review.reviewer_available else "error",
                              output=f"score {review.score}/10, approved={review.approved}",
                              confidence=review.confidence)}


def confidence_check(state: AgentState) -> dict:
    settings = get_settings()
    review = state.get("review")
    subtasks = _subtasks(state)
    breakdown = compute_confidence(review, subtasks, state.get("results") or {}, state.get("tool_calls") or [],
                                   state.get("human_decisions") or [])
    t = NodeTrace(_trace_id(state), "confidence_check", "supervisor")
    notes: list[str] = []
    revisions = state.get("revision_count") or 0
    if (review and review.reviewer_available and not review.approved and review.revision_targets
            and revisions < settings.max_revisions):
        route = "revise"
    elif breakdown.tier in ("auto", "notify"):
        route = "synthesis"
        if breakdown.tier == "notify":
            notes.append(f"Confidence {breakdown.score:.0f}/100 is in the notify band; continued automatically. "
                         "Check the reviewer notes.")
    else:
        route = "human_review"
    t.info("confidence", breakdown.model_dump())
    return {"confidence": breakdown, "route": route, "notifications": notes,
            "trace": t.events(output=f"{breakdown.score} ({breakdown.tier}) -> {route}",
                              confidence=breakdown.score / 100)}


def revise(state: AgentState) -> dict:
    review = state["review"]
    subtasks = _subtasks(state)
    results = state.get("results") or {}
    targets = {sid for sid in review.revision_targets
               if not (results.get(sid) and results[sid].status == "human")}
    affected = _dependents(subtasks, targets)
    feedback = review.feedback + ("\nIssues: " + "; ".join(review.issues) if review.issues else "")
    updates = {}
    for sub in subtasks:
        if sub.subtask_id in affected and not (results.get(sub.subtask_id) and results[sub.subtask_id].status == "human"):
            note = feedback if sub.subtask_id in targets else (
                f"Upstream subtask(s) {', '.join(sorted(targets))} were revised; update your output accordingly.")
            updates[sub.subtask_id] = sub.model_copy(update={"status": "pending", "revision_feedback": note,
                                                             "last_error": None})
    t = NodeTrace(_trace_id(state), "revise", "supervisor")
    t.info("revise", f"Revising {sorted(updates)} after review score {review.score}/10")
    return {"subtasks": updates, "revision_count": (state.get("revision_count") or 0) + 1,
            "notifications": [f"Reviewer requested revision of {', '.join(sorted(targets))}."],
            "trace": t.events(output=f"{len(updates)} subtask(s) sent back")}


def human_review(state: AgentState) -> dict:
    settings = get_settings()
    breakdown = state["confidence"]
    review = state.get("review")
    subtasks = _subtasks(state)
    results = state.get("results") or {}
    terminal = _terminal_ids(subtasks)
    draft = "\n\n".join(results[s].output for s in terminal if s in results and results[s].output)
    level = "approve_action" if breakdown.tier == "approve_action" else "take_over"
    used = set(state.get("memories_used") or [])
    request = ApprovalRequest(
        request_id=f"final-{len(state.get('review_history') or [])}", checkpoint="final_output", level=level,
        task=state["task"].user_request, agent="supervisor",
        proposed_action="Synthesize and deliver the final answer from the reviewed outputs",
        reason=(f"Confidence {breakdown.score:.0f}/100 is below the automatic threshold "
                f"({settings.auto_threshold:.0f}). Reviewer: {review.feedback if review else 'no review'}"),
        confidence=breakdown.score,
        relevant_memories=[m for m in state.get("retrieved_memories") or [] if m.memory_id in used],
        details=draft or "(no draft output available)",
        options_help={"approve": "Approve: deliver the answer as-is", "modify": "Modify: send your instructions "
                      "back to the agents for another pass", "reject": "Reject: stop without an answer",
                      "take_over": "Take Over: write the final answer yourself"},
    )
    value = interrupt(request.model_dump(mode="json"))
    decision = _decision(value, checkpoint="final_output", level=level, subject=request.proposed_action)
    t = NodeTrace(_trace_id(state), "human_review", "human")
    t.human(decision)
    update: dict[str, Any] = {"human_decisions": [decision], "trace": t.events(output=decision.decision)}
    if decision.feedback:
        update["human_notes"] = [decision.feedback]
    if decision.decision == "approve":
        update["route"] = "synthesis"
    elif decision.decision == "take_over":
        update.update(route="synthesis", human_final_override=decision.modified_instruction)
    elif decision.decision == "reject":
        update["route"] = "abort"
    else:
        targets = set(review.revision_targets if review and review.revision_targets else terminal)
        affected = _dependents(subtasks, targets)
        update["subtasks"] = {
            s.subtask_id: s.model_copy(update={"status": "pending", "human_instruction": decision.modified_instruction})
            for s in subtasks if s.subtask_id in affected
        }
        update["human_notes"] = [*update.get("human_notes", []), decision.modified_instruction]
        update["route"] = "dispatch"
    return update


def synthesis(state: AgentState) -> dict:
    task = state["task"]
    t = NodeTrace(task.trace_id, "synthesis", "supervisor")
    subtasks = _subtasks(state)
    results = state.get("results") or {}
    breakdown = state.get("confidence")
    human_verified = bool(breakdown and breakdown.human_verified) or any(
        d.checkpoint == "final_output" and d.decision == "approve" for d in state.get("human_decisions") or [])
    notes: list[str] = []
    all_sources = {}
    for r in results.values():
        for s in r.sources:
            all_sources.setdefault(s.source_id, s)
    agents_used = sorted({r.agent_name for r in results.values() if r.status != "failed"})
    tools_used = sorted({c.tool for c in state.get("tool_calls") or [] if c.status == "success"})
    override = state.get("human_final_override")
    summary = ""
    if override:
        final = FinalResult(answer=override, confidence=breakdown.score if breakdown else 0.0,
                            agents_used=agents_used, tools_used=tools_used, human_verified=True, produced_by="human")
        t.info("synthesis.human_override", "Final answer provided by a human reviewer.")
    else:
        usable = [r for r in results.values() if r.status != "failed" and r.output]
        if not usable:
            t.error("synthesis", "No subtask produced usable output.")
            return {"final": None, "notifications": ["No subtask produced usable output; no answer was generated."],
                    "trace": t.events(status="error", error="no usable output")}
        produced_by = "supervisor"
        try:
            out, info = sup.synthesize(task.user_request, state["intake"].user_preferences if state.get("intake") else [],
                                       subtasks, results, state.get("review"), state.get("human_notes") or [])
            t.llm("supervisor.synthesize", info, output=out.answer[:800])
            answer, summary = out.answer, out.summary
        except LLMError as exc:
            t.error("supervisor.synthesize", exc, user_message=exc.user_message)
            terminal = [results[s] for s in _terminal_ids(subtasks) if s in results and results[s].output]
            answer = "\n\n".join(r.output for r in (terminal or usable))
            produced_by = "writer_fallback"
            notes.append("Final synthesis failed; showing the Writer's reviewed output instead.")
        answer, sources = sup.renumber_citations(answer, list(all_sources.values()))
        final = FinalResult(answer=answer, confidence=breakdown.score if breakdown else 0.0, sources=sources,
                            agents_used=agents_used, tools_used=tools_used, human_verified=human_verified,
                            produced_by=produced_by)  # type: ignore[arg-type]
    return {"final": final, "final_summary": summary, "notifications": notes,
            "trace": t.events(output=final.answer[:600], confidence=final.confidence / 100)}


def finalize(state: AgentState) -> dict:
    settings = get_settings()
    task = state["task"]
    t = NodeTrace(task.trace_id, "finalize", "system")
    final = state.get("final")
    aborted = state.get("route") == "abort" or state.get("plan_status") == "rejected"
    config_failure = aborted and not state.get("intake")
    if config_failure:
        status = "failed"
    elif aborted:
        status = "rejected"
    else:
        status = "completed" if final else "failed"

    stored: list[MemoryRecord] = []
    notes: list[str] = []
    if settings.memory_enabled and not config_failure:
        candidates: list[tuple[str, str, dict]] = []
        intake = state.get("intake")
        for pref in intake.user_preferences if intake else []:
            candidates.append((f"User preference: {pref}", "user_preference", {}))
        for d in state.get("human_decisions") or []:
            text = d.modified_instruction or d.feedback
            if d.decision in ("modify", "take_over", "reject") and text:
                candidates.append((f"Human {d.decision.replace('_', ' ')} at {d.checkpoint} for task "
                                   f"'{task.user_request[:150]}': {text[:600]}", "human_correction",
                                   {"checkpoint": d.checkpoint, "decision": d.decision}))
        breakdown = state.get("confidence")
        good = status == "completed" and final and (
            final.human_verified or (breakdown and breakdown.score >= settings.memory_min_confidence))
        if good:
            plan = " -> ".join(f"{s.assigned_agent}: {s.description[:80]}" for s in _subtasks(state))
            score = breakdown.score if breakdown else 0
            candidates.append((f"Successful strategy for '{task.user_request[:200]}' (confidence {score:.0f}): {plan}",
                               "successful_strategy", {"confidence": float(score)}))
            if state.get("final_summary"):
                candidates.append((f"Completed task '{task.user_request[:200]}': {state['final_summary']}",
                                   "task_decision", {"confidence": float(score)}))
        if candidates:
            try:
                from memory.long_term import get_long_term_memory

                store = get_long_term_memory()
                for content, mtype, meta in candidates:
                    started = time.perf_counter()
                    record, reason = store.add(content, mtype, task.task_id, meta)  # type: ignore[arg-type]
                    t.memory(f"long_term.add.{mtype}", input=content[:300], output=reason,
                             latency_ms=(time.perf_counter() - started) * 1000,
                             status="success" if record else "skipped")
                    if record:
                        stored.append(record)
            except Exception as exc:  # noqa: BLE001
                t.error("long_term.add", exc, user_message="Memory could not be saved.")
                notes.append("Long-term memory could not be updated for this task.")

    task = task.model_copy(update={"status": status, "completed_at": utcnow()})
    events = t.events(output=f"status={status}; {len(stored)} memories stored")
    decisions = state.get("human_decisions") or []
    human_time = sum(d.review_seconds for d in decisions)
    elapsed = max(0.0, time.time() - (state.get("started_at") or time.time()) - human_time)
    metrics = compute_metrics([*(state.get("trace") or []), *events], decisions, elapsed)
    update: dict[str, Any] = {"task": task, "stored_memories": stored, "metrics": metrics, "trace": events,
                              "notifications": notes}
    if final:
        update["final"] = final.model_copy(update={"execution_time": metrics.execution_time_s})
    return update


# =========================================================================================
# routing
# =========================================================================================
def route_after_intake(state: AgentState) -> str:
    return "finalize" if state.get("route") == "abort" else "memory_retrieval"


def route_after_plan_approval(state: AgentState) -> str:
    return {"approved": "specialist_dispatch", "modify": "supervisor", "rejected": "finalize",
            "taken_over": "synthesis"}.get(state.get("plan_status", "approved"), "specialist_dispatch")


def _send_wave(state: AgentState, node: str) -> list[Send]:
    return [Send(node, {**state, "current_subtask_id": sid}) for sid in state.get("wave") or []]


def route_after_dispatch(state: AgentState) -> list[Send] | str:
    return _send_wave(state, "tool_planning") if state.get("wave") else "reviewer"


def route_after_action_approval(state: AgentState) -> list[Send] | str:
    if _pending_action(state) is not None:
        return "action_approval"
    return _send_wave(state, "specialist_execution")


def route_after_collection(state: AgentState) -> str:
    return "failure_escalation" if state.get("escalations") else "specialist_dispatch"


def route_after_escalation(state: AgentState) -> str:
    if state.get("route") == "abort":
        return "finalize"
    return "failure_escalation" if state.get("escalations") else "specialist_dispatch"


def route_after_confidence(state: AgentState) -> str:
    return state.get("route", "synthesis")


def route_after_human_review(state: AgentState) -> str:
    return {"synthesis": "synthesis", "dispatch": "specialist_dispatch", "abort": "finalize"}[state["route"]]


def _serializer() -> JsonPlusSerializer:
    """Allow-list exactly our Pydantic models for checkpoint (de)serialization."""
    allowed = []
    for module in (schemas, short_term):
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseModel) and obj.__module__ == module.__name__:
                allowed.append((module.__name__, name))
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("task_intake", task_intake)
    graph.add_node("memory_retrieval", memory_retrieval)
    graph.add_node("supervisor", supervisor_plan)
    graph.add_node("plan_validation", plan_validation)
    graph.add_node("plan_approval", plan_approval)
    graph.add_node("specialist_dispatch", specialist_dispatch)
    graph.add_node("tool_planning", tool_planning)
    graph.add_node("action_approval", action_approval)
    graph.add_node("specialist_execution", specialist_execution)
    graph.add_node("result_collection", result_collection)
    graph.add_node("failure_escalation", failure_escalation)
    graph.add_node("reviewer", reviewer)
    graph.add_node("confidence_check", confidence_check)
    graph.add_node("revise", revise)
    graph.add_node("human_review", human_review)
    graph.add_node("synthesis", synthesis)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "task_intake")
    graph.add_conditional_edges("task_intake", route_after_intake, ["memory_retrieval", "finalize"])
    graph.add_edge("memory_retrieval", "supervisor")
    graph.add_edge("supervisor", "plan_validation")
    graph.add_edge("plan_validation", "plan_approval")
    graph.add_conditional_edges("plan_approval", route_after_plan_approval,
                                ["specialist_dispatch", "supervisor", "finalize", "synthesis"])
    graph.add_conditional_edges("specialist_dispatch", route_after_dispatch, ["tool_planning", "reviewer"])
    graph.add_edge("tool_planning", "action_approval")
    graph.add_conditional_edges("action_approval", route_after_action_approval,
                                ["action_approval", "specialist_execution"])
    graph.add_edge("specialist_execution", "result_collection")
    graph.add_conditional_edges("result_collection", route_after_collection,
                                ["failure_escalation", "specialist_dispatch"])
    graph.add_conditional_edges("failure_escalation", route_after_escalation,
                                ["failure_escalation", "specialist_dispatch", "finalize"])
    graph.add_edge("reviewer", "confidence_check")
    graph.add_conditional_edges("confidence_check", route_after_confidence, ["revise", "synthesis", "human_review"])
    graph.add_edge("revise", "specialist_dispatch")
    graph.add_conditional_edges("human_review", route_after_human_review,
                                ["synthesis", "specialist_dispatch", "finalize"])
    graph.add_edge("synthesis", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=InMemorySaver(serde=_serializer()))


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


# =========================================================================================
# runner (used by the Streamlit app)
# =========================================================================================
@dataclass
class Progress:
    kind: str  # "step" | "interrupt" | "error" | "done"
    message: str
    node: str = ""
    ok: bool = True


def _config(task_id: str) -> dict:
    return {"configurable": {"thread_id": task_id}, "recursion_limit": get_settings().recursion_limit}


def _describe(node: str, update: dict | None) -> tuple[str, bool]:
    u = update or {}
    if node == "task_intake":
        intake = u.get("intake")
        return (f"Task received ({intake.complexity})" if intake else "Task intake failed", intake is not None)
    if node == "memory_retrieval":
        return f"Long-term memory searched: {len(u.get('retrieved_memories') or [])} relevant memories", True
    if node == "supervisor":
        return f"Supervisor created plan ({len(u['raw_plan'].subtasks)} subtasks)", True
    if node == "plan_validation":
        return f"Plan validated ({len(u.get('plan_issues') or [])} adjustments)", True
    if node == "plan_approval":
        return f"Plan {u.get('plan_status', 'approved')}", True
    if node == "specialist_dispatch":
        wave = u.get("wave") or []
        return (f"Dispatching {', '.join(wave)}" + (" in parallel" if len(wave) > 1 else "")
                if wave else "All subtasks finished"), True
    if node == "tool_planning":
        (sid, plan), = u["tool_plans"].items()
        tools = ", ".join(c.tool for c in plan.calls) or "no tools"
        return f"{AGENT_LABELS[plan.agent]} planned {sid}: {tools}", True
    if node == "action_approval":
        return ("Tool call decision recorded" if u.get("human_decisions") else "Tool calls cleared for execution"), True
    if node == "specialist_execution":
        (sid, result), = u["results"].items()
        ok = result.status == "success"
        return (f"{AGENT_LABELS.get(result.agent_name, result.agent_name)} "
                f"{'completed' if ok else 'failed'} {sid}"), ok
    if node == "result_collection":
        return "Results collected", not u.get("escalations")
    if node == "failure_escalation":
        return "Failure escalation resolved", True
    if node == "reviewer":
        review = u["review"]
        return f"Reviewer evaluated: {review.score:.1f}/10, {'approved' if review.approved else 'changes requested'}", \
            review.approved
    if node == "confidence_check":
        c = u["confidence"]
        return f"Confidence {c.score:.0f}/100 ({c.tier.replace('_', ' ')})", c.tier in ("auto", "notify")
    if node == "revise":
        return "Revision requested by Reviewer", False
    if node == "human_review":
        return "Human review decision recorded", True
    if node == "synthesis":
        return ("Supervisor synthesized the final answer" if u.get("final") else "No answer could be synthesized",
                bool(u.get("final")))
    if node == "finalize":
        return f"Run finished: {u['task'].status}", u["task"].status == "completed"
    return node, True


def _drive(payload: Any, task_id: str) -> Iterator[Progress]:
    graph = get_graph()
    config = _config(task_id)
    failed_message: str | None = None
    try:
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    yield Progress("interrupt", "Human approval is required before this action can continue.", node)
                else:
                    message, ok = _describe(node, update)
                    yield Progress("step", message, node, ok)
    except Exception as exc:  # noqa: BLE001 - never let a workflow error crash the UI
        log.exception("Workflow error for task %s", task_id)
        failed_message = getattr(exc, "user_message", None) or (
            "An unexpected error stopped the workflow. Technical details were written to data/app.log.")
        yield Progress("error", failed_message, ok=False)
    finally:
        _persist(task_id, failed_message)
    if failed_message is None and pending_approval(task_id) is None:
        yield Progress("done", "Workflow complete.")


def start_run(task_id: str, user_request: str, uploads: list[UploadedFile], options: RunOptions) -> Iterator[Progress]:
    task = Task(task_id=task_id, trace_id=new_id("trace"), user_request=user_request.strip())
    initial = {"task": task, "options": options, "uploads": uploads, "started_at": time.time(),
               "revision_count": 0, "plan_version": 0}
    yield from _drive(initial, task_id)


def resume_run(task_id: str, decision: dict) -> Iterator[Progress]:
    yield from _drive(Command(resume=decision), task_id)


def pending_approval(task_id: str) -> ApprovalRequest | None:
    try:
        snapshot = get_graph().get_state(_config(task_id))
    except Exception:  # noqa: BLE001
        return None
    for item in getattr(snapshot, "interrupts", ()) or ():
        return ApprovalRequest.model_validate(item.value)
    return None


def current_snapshot(task_id: str) -> RunSnapshot | None:
    try:
        values = get_graph().get_state(_config(task_id)).values
    except Exception:  # noqa: BLE001
        return None
    if not values or "task" not in values:
        return None
    snap = snapshot_from_state(values)
    if pending_approval(task_id) is not None and snap.task.status == "running":
        snap.task = snap.task.model_copy(update={"status": "awaiting_human"})
    return snap


def _persist(task_id: str, failed_message: str | None) -> None:
    from core.storage import save_run

    try:
        values = get_graph().get_state(_config(task_id)).values
        if not values or "task" not in values:
            return
        snapshot = snapshot_from_state(values)
        if failed_message:
            snapshot.task = snapshot.task.model_copy(update={"status": "failed", "completed_at": utcnow()})
            snapshot.notifications.append(failed_message)
        elif pending_approval(task_id) is not None:
            snapshot.task = snapshot.task.model_copy(update={"status": "awaiting_human"})
        metrics = snapshot.metrics
        if metrics is None:
            human_time = sum(d.review_seconds for d in snapshot.human_decisions)
            elapsed = time.time() - (values.get("started_at") or time.time()) - human_time
            metrics = compute_metrics(snapshot.trace, snapshot.human_decisions, max(0.0, elapsed))
            snapshot.metrics = metrics
        save_run(snapshot, metrics)
    except Exception:  # noqa: BLE001 - persistence failures are logged, not surfaced as crashes
        log.exception("Failed to persist task %s", task_id)
