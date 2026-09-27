"""Streamlit renderers for a RunSnapshot.

The same functions render the live task and replayed tasks, so what you see on replay is
exactly what the run recorded.
"""

from __future__ import annotations

import json
from collections import defaultdict

import streamlit as st

from core.config import get_settings
from core.schemas import AGENT_LABELS, TraceEvent
from memory.short_term import RunSnapshot

STATUS_ICON = {"pending": "⏳ pending", "running": "⟳ running", "completed": "✅ completed", "failed": "❌ failed",
               "skipped": "⏭ skipped"}
TASK_STATUS = {"running": "⟳ Running", "awaiting_human": "⏸ Awaiting human", "completed": "✅ Completed",
               "rejected": "🛑 Rejected", "failed": "❌ Failed"}
TIER_LABEL = {"auto": "Automatic continuation", "notify": "Notify", "approve_action": "Approve action",
              "escalate": "Human escalation"}


def _label(agent: str) -> str:
    return AGENT_LABELS.get(agent, agent)


def _short(value, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + " …"


# ---------------------------------------------------------------------------------------
def render_progress_from_trace(snap: RunSnapshot) -> None:
    """Execution steps reconstructed from node events (used when no live log is available)."""
    for event in snap.trace:
        if event.event_type != "node":
            continue
        icon = {"success": "✓", "error": "✗", "skipped": "⏭", "waiting": "⏸"}.get(event.status, "•")
        st.markdown(f"{icon} **{_label(event.agent)}** · `{event.node}` — {_short(event.output or '', 140)}")


def render_notifications(snap: RunSnapshot) -> None:
    if snap.notifications:
        with st.container(border=True):
            st.markdown("**Notifications**")
            for note in dict.fromkeys(snap.notifications):
                st.markdown(f"- {note}")


def render_plan(snap: RunSnapshot) -> None:
    if not snap.subtasks:
        st.caption("No plan yet.")
        return
    if snap.plan_reasoning:
        st.caption(f"Supervisor reasoning: {snap.plan_reasoning}")
    rows = []
    for sub in snap.subtasks:
        result = snap.results.get(sub.subtask_id)
        rows.append({
            "Subtask": sub.subtask_id,
            "Description": sub.description,
            "Assigned Agent": _label(result.agent_name if result else sub.assigned_agent),
            "Dependencies": ", ".join(sub.dependencies) or "—",
            "Parallel group": sub.parallel_group,
            "Status": STATUS_ICON.get(sub.status, sub.status),
            "Confidence": round(result.confidence * 100) if result and result.status != "failed" else None,
            "Attempts": sub.attempts,
            "Notes": "; ".join(filter(None, [
                f"reassigned from {sub.reassigned_from}" if sub.reassigned_from else "",
                "human instruction" if sub.human_instruction else "",
                "revised" if sub.revision_feedback else "",
            ])),
        })
    st.dataframe(rows, hide_index=True, column_config={
        "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d"),
    })
    if snap.plan_issues:
        with st.expander(f"Plan validation adjustments ({len(snap.plan_issues)})"):
            for issue in snap.plan_issues:
                st.markdown(f"- {issue}")
    if len(snap.plan_history) > 1:
        st.caption(f"Plan revised {len(snap.plan_history) - 1} time(s) after human feedback.")


def render_final(snap: RunSnapshot) -> None:
    final = snap.final
    if final is None:
        if snap.task.status == "rejected":
            st.error("The task was rejected by the human reviewer; no answer was produced.")
        elif snap.task.status == "failed":
            st.error("No final answer could be produced. See notifications and the execution trace.")
        else:
            st.caption("The final answer appears here when the run completes.")
        return
    badges = []
    if final.produced_by == "human":
        badges.append("✍️ Human-provided")
    elif final.produced_by == "writer_fallback":
        badges.append("⚠️ Writer output (synthesis unavailable)")
    if final.human_verified:
        badges.append("🧑‍⚖️ Human-verified")
    if badges:
        st.markdown(" · ".join(badges))
    with st.container(border=True):
        st.markdown(final.answer)
    if final.sources:
        st.markdown("**Sources** (retrieved by the web_search tool)")
        for i, src in enumerate(final.sources, start=1):
            st.markdown(f"{i}. [{src.title or src.source}]({src.url}) — {src.source}")
    cols = st.columns(4)
    cols[0].metric("Confidence", f"{final.confidence:.0f}/100")
    cols[1].metric("Execution time", f"{final.execution_time:.1f}s")
    cols[2].metric("Agents used", len(final.agents_used))
    cols[3].metric("Tools used", len(final.tools_used))
    st.caption(f"Agents: {', '.join(_label(a) for a in final.agents_used) or '—'} · "
               f"Tools: {', '.join(final.tools_used) or '—'}")


def render_confidence(snap: RunSnapshot) -> None:
    conf = snap.confidence
    if conf is None:
        st.caption("Confidence is computed after the Reviewer runs.")
        return
    settings = get_settings()
    st.markdown(f"**{conf.score:.1f}/100** → {TIER_LABEL[conf.tier]}"
                + (" · human-verified" if conf.human_verified else ""))
    signals = ["review_score", "task_completion", "tool_success", "agent_agreement", "information_completeness"]
    total_w = sum(conf.weights.values()) or 1
    rows = [{
        "Signal": s.replace("_", " ").title(), "Value (0-1)": getattr(conf, s), "Weight": conf.weights.get(s, 0),
        "Contribution": round(100 * conf.weights.get(s, 0) * getattr(conf, s) / total_w, 1),
    } for s in signals]
    st.dataframe(rows, hide_index=True, column_config={
        "Value (0-1)": st.column_config.ProgressColumn("Value", min_value=0, max_value=1, format="%.2f"),
    })
    st.caption(f"Thresholds: ≥{settings.auto_threshold:.0f} automatic · ≥{settings.notify_threshold:.0f} notify · "
               f"≥{settings.approve_threshold:.0f} approve action · below that, human escalation.")
    for note in conf.notes:
        st.markdown(f"- {note}")


def render_metrics(snap: RunSnapshot) -> None:
    m = snap.metrics
    if m is None:
        return
    row1 = st.columns(4)
    row1[0].metric("Input tokens", f"{m.input_tokens:,}")
    row1[1].metric("Output tokens", f"{m.output_tokens:,}")
    row1[2].metric("Total tokens", f"{m.total_tokens:,}")
    row1[3].metric("Cost", m.cost_label)
    row2 = st.columns(4)
    row2[0].metric("LLM calls", m.llm_calls)
    row2[1].metric("Tool calls", m.tool_calls)
    row2[2].metric("Execution time", f"{m.execution_time_s:.1f}s")
    row2[3].metric("Human review time", f"{m.human_review_time_s:.0f}s")


def render_agent_activity(snap: RunSnapshot) -> None:
    with st.expander("Supervisor"):
        if snap.intake:
            st.markdown(f"**Intake:** {snap.intake.complexity} — {snap.intake.reasoning}")
            if snap.intake.user_preferences:
                st.markdown("**Detected preferences:** " + "; ".join(snap.intake.user_preferences))
        st.markdown(f"**Plan reasoning:** {snap.plan_reasoning or '—'}")
        st.markdown(f"**Plan versions:** {len(snap.plan_history)} · **Memories used in planning:** "
                    f"{', '.join(snap.memories_used) or 'none'}")
        decisions = [e for e in snap.trace if e.agent == "supervisor" and e.event_type in ("llm_call", "retry", "error")]
        for e in decisions:
            st.markdown(f"- `{e.name}` · {e.status} · {e.latency_ms:.0f} ms"
                        + (f" · {e.token_usage.total_tokens} tokens" if e.token_usage else "")
                        + (f" — {e.error}" if e.error else ""))
        if snap.final:
            st.markdown(f"**Final answer produced by:** {snap.final.produced_by}")

    for agent in ("researcher", "analyst", "writer"):
        results = [r for r in snap.results.values() if r.agent_name == agent]
        plans = [p for p in snap.tool_plans if p.agent == agent]
        with st.expander(f"{_label(agent)} ({len(results)} result{'s' if len(results) != 1 else ''})"):
            if not results and not plans:
                st.caption("Not used in this run.")
            for plan in plans:
                st.markdown(f"**Tool plan for {plan.subtask_id}** (confidence {plan.confidence:.2f}): {plan.reasoning}")
                for call in plan.calls:
                    st.markdown(f"- `{call.tool}` {_short(call.arguments, 120)} — {call.reason} "
                                f"[{call.approval.replace('_', ' ')}]")
            for r in results:
                st.divider()
                st.markdown(f"**{r.subtask_id}** · {r.status} · confidence {r.confidence:.2f} · {r.latency:.1f}s · "
                            f"{r.token_usage.total_tokens:,} tokens · attempts {r.attempts} · "
                            f"tools: {', '.join(r.tools_used) or 'none'}")
                if r.error:
                    st.error(r.error)
                if r.output:
                    st.markdown(r.output)
                if r.limitations:
                    st.caption("Limitations: " + "; ".join(r.limitations))

    human_results = [r for r in snap.results.values() if r.agent_name == "human"]
    if human_results:
        with st.expander("Human (take-over outputs)"):
            for r in human_results:
                st.markdown(f"**{r.subtask_id}**\n\n{r.output}")

    with st.expander(f"Reviewer ({len(snap.review_history)} review{'s' if len(snap.review_history) != 1 else ''})"):
        if not snap.review_history:
            st.caption("No review yet.")
        for i, rv in enumerate(snap.review_history, start=1):
            st.markdown(f"**Review {i}:** {'✅ approved' if rv.approved else '↩️ changes requested'} · "
                        f"score {rv.score:.1f}/10 · consistency {rv.consistency_score:.1f}/10 · "
                        f"confidence {rv.confidence:.2f}")
            st.markdown(f"_Feedback:_ {rv.feedback}")
            for title, items in (("Issues", rv.issues), ("Missing information", rv.missing_information),
                                 ("Unsupported claims", rv.unsupported_claims), ("Strengths", rv.strengths)):
                if items:
                    st.markdown(f"**{title}:** " + "; ".join(items))
            if rv.revision_targets:
                st.markdown(f"**Sent back for revision:** {', '.join(rv.revision_targets)}")
            st.divider()


def render_tool_usage(snap: RunSnapshot, key: str = "live") -> None:
    if not snap.tool_calls:
        st.caption("No tool calls in this run.")
        return
    rows = [{
        "Tool": c.tool, "Agent": _label(c.agent), "Subtask": c.subtask_id, "Input": _short(c.input, 120),
        "Output": _short(c.output if c.output is not None else (c.error or ""), 160),
        "Latency (ms)": round(c.latency_ms), "Status": c.status,
        "Human approval": {True: "approved", False: "denied", None: "—"}[c.approved_by_human],
    } for c in snap.tool_calls]
    st.dataframe(rows, hide_index=True)
    options = {f"{c.tool} · {c.subtask_id} · {c.call_id}": c for c in snap.tool_calls}
    choice = st.selectbox("Inspect a tool call", list(options), key=f"tool_inspect_{key}_{snap.task.task_id}")
    if choice:
        call = options[choice]
        st.json({"input": call.input, "output": call.output, "error": call.error, "status": call.status,
                 "tokens": call.token_usage.model_dump(), "llm_calls": call.llm_calls})


def render_memory_current(snap: RunSnapshot) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Current Task Context")
        st.caption("Short-term memory: the LangGraph state for this task.")
        st.markdown(f"**Task:** {snap.task.user_request}")
        st.markdown(f"**Status:** {TASK_STATUS.get(snap.task.status, snap.task.status)} · "
                    f"**Complexity:** {snap.task.complexity or '—'}")
        if snap.uploads:
            st.markdown("**Uploaded files:** " + ", ".join(u.file_name for u in snap.uploads))
        st.markdown(f"**Subtasks:** {len(snap.subtasks)} · **Agent outputs:** {len(snap.results)} · "
                    f"**Tool results:** {len(snap.tool_calls)} · **Reviews:** {len(snap.review_history)} · "
                    f"**Human decisions:** {len(snap.human_decisions)}")
        if snap.review:
            st.markdown(f"**Latest review feedback:** {snap.review.feedback}")
        for d in snap.human_decisions:
            st.markdown(f"- Human **{d.decision}** at {d.checkpoint}"
                        + (f": {d.modified_instruction or d.feedback}" if (d.modified_instruction or d.feedback) else ""))
    with right:
        st.markdown("#### Retrieved Long-Term Memories")
        st.caption("Semantic matches from ChromaDB, retrieved before planning.")
        if not snap.retrieved_memories:
            st.caption("No relevant long-term memories were found for this task.")
        used = set(snap.memories_used)
        for m in snap.retrieved_memories:
            tag = "✅ used by Supervisor" if m.memory_id in used else "ignored by Supervisor"
            with st.container(border=True):
                st.markdown(f"`{m.memory_type}` · distance {m.distance} · {tag}")
                st.markdown(m.content)
        if snap.stored_memories:
            st.markdown("#### Saved to long-term memory by this run")
            for m in snap.stored_memories:
                st.markdown(f"- `{m.memory_type}` {m.content}")


def render_human_decisions(snap: RunSnapshot) -> None:
    if not snap.human_decisions:
        st.caption("No human decisions in this run.")
        return
    st.dataframe([{
        "Checkpoint": d.checkpoint, "Level": d.approval_level, "Decision": d.decision, "Subject": d.subject,
        "Instruction / output": d.modified_instruction, "Feedback": d.feedback, "Review time (s)": round(d.review_seconds, 1),
    } for d in snap.human_decisions], hide_index=True)


# ---------------------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------------------
def _event_line(e: TraceEvent) -> str:
    bits = [f"{e.event_type}: {e.name}"]
    if e.latency_ms:
        bits.append(f"{e.latency_ms:.0f} ms")
    if e.token_usage:
        bits.append(f"{e.token_usage.total_tokens} tok")
    if e.human_decision:
        bits.append(f"human: {e.human_decision}")
    if e.status not in ("success", "info"):
        bits.append(e.status.upper())
    return " · ".join(bits)


def agent_tree(trace: list[TraceEvent]) -> str:
    """Supervisor-rooted tree of agents and the tools/memory/LLM calls each performed."""
    by_agent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    order: list[str] = []
    for e in trace:
        if e.event_type == "node":
            continue
        agent = e.agent
        if agent not in order:
            order.append(agent)
        key = {"tool_call": f"Tool: {e.tool}", "memory": "Memory Retrieval" if "search" in e.name else "Memory Write",
               "llm_call": "LLM call", "human": "Human decision", "retry": "Retry", "error": "Error"}.get(e.event_type)
        if key:
            by_agent[agent][key] += 1
    lines = ["Supervisor"]
    children = [a for a in order if a != "supervisor"]
    sup_items = by_agent.get("supervisor", {})
    entries: list[tuple[str, dict[str, int]]] = [("(supervisor actions)", sup_items)] if sup_items else []
    entries += [(_label(a), by_agent[a]) for a in children]
    for i, (name, items) in enumerate(entries):
        last = i == len(entries) - 1
        lines.append(f"{'└──' if last else '├──'} {name}")
        for j, (item, count) in enumerate(items.items()):
            lines.append(f"{'    ' if last else '│   '}{'└──' if j == len(items) - 1 else '├──'} {item} ×{count}")
    return "\n".join(lines)


def timeline_tree(trace: list[TraceEvent]) -> str:
    roots = [e for e in trace if e.event_type == "node"]
    children: dict[str, list[TraceEvent]] = defaultdict(list)
    for e in trace:
        if e.parent_id:
            children[e.parent_id].append(e)
    lines = []
    for i, root in enumerate(roots):
        last = i == len(roots) - 1
        status = "" if root.status in ("success", "info") else f" [{root.status}]"
        lines.append(f"{'└──' if last else '├──'} {_label(root.agent)} · {root.node} ({root.latency_ms:.0f} ms){status}")
        kids = children.get(root.event_id, [])
        for j, kid in enumerate(kids):
            lines.append(f"{'    ' if last else '│   '}{'└──' if j == len(kids) - 1 else '├──'} {_event_line(kid)}")
    return "\n".join(lines)


def render_trace(snap: RunSnapshot, key: str = "live") -> None:
    if not snap.trace:
        st.caption("No trace events yet.")
        return
    st.markdown(f"**Trace ID:** `{snap.task.trace_id}` · **Task ID:** `{snap.task.task_id}` · "
                f"{len(snap.trace)} events")
    render_metrics(snap)
    view = st.radio("View", ["Agent tree", "Timeline tree", "Event explorer"], horizontal=True, key=f"trace_view_{key}")
    if view == "Agent tree":
        st.code(agent_tree(snap.trace), language="text")
    elif view == "Timeline tree":
        st.code(timeline_tree(snap.trace), language="text")
    else:
        types = sorted({e.event_type for e in snap.trace})
        chosen = st.multiselect("Event types", types, default=types, key=f"trace_types_{key}")
        roots = [e for e in snap.trace if e.event_type == "node"]
        children: dict[str, list[TraceEvent]] = defaultdict(list)
        for e in snap.trace:
            if e.parent_id:
                children[e.parent_id].append(e)
        for idx, root in enumerate(roots):
            kids = [k for k in children.get(root.event_id, []) if k.event_type in chosen]
            if "node" not in chosen and not kids:
                continue
            icon = {"success": "✓", "error": "✗", "skipped": "⏭"}.get(root.status, "•")
            with st.expander(f"{icon} {idx + 1}. {_label(root.agent)} · {root.node} · {root.latency_ms:.0f} ms"
                             + (f" · {root.token_usage.total_tokens} tok" if root.token_usage else "")):
                st.json(root.model_dump(mode="json", exclude_none=True), expanded=False)
                for kid in kids:
                    st.markdown(f"**↳ {_event_line(kid)}**")
                    st.json(kid.model_dump(mode="json", exclude_none=True), expanded=False)
