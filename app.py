"""Streamlit entry point for the Agent Orchestration System.

    streamlit run app.py

Session state (st.session_state) holds only UI state: the current task id, the live progress
log, when each approval request was first shown (to measure human review time) and which
approval mode (modify / take over) is open. The workflow state itself lives in the LangGraph
checkpointer, keyed by task id. Completed and paused runs are persisted to SQLite.
"""

from __future__ import annotations

import time

import streamlit as st

st.set_page_config(page_title="AI Agent Orchestrator", page_icon="🧭", layout="wide")

from core.config import get_settings, setup_logging  # noqa: E402
from core.llm import check_connection  # noqa: E402
from core.schemas import AGENT_LABELS, ApprovalRequest, RunOptions, new_id  # noqa: E402
from core.storage import analytics, init_db, list_tasks, load_snapshot  # noqa: E402
from graph import workflow  # noqa: E402
from memory.short_term import RunSnapshot  # noqa: E402
from tools.file_tools import SUPPORTED_TYPES, save_uploads  # noqa: E402
from tools.registry import REGISTRY  # noqa: E402
from ui import views  # noqa: E402

setup_logging()
settings = get_settings()

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1400px;}
      div[data-testid="stMetricValue"] {font-size: 1.35rem;}
      .hitl-banner {border-left: 4px solid #f59e0b; padding: .6rem 1rem; background: rgba(245,158,11,.08);
                    border-radius: 4px; margin-bottom: .75rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _bootstrap() -> bool:
    init_db()
    return True


@st.cache_data(ttl=300, show_spinner=False)
def _connection_status() -> tuple[bool, str]:
    return check_connection()


def _memory_count() -> int | None:
    if not settings.memory_enabled:
        return None
    try:
        from memory.long_term import get_long_term_memory

        return get_long_term_memory().count()
    except Exception:  # noqa: BLE001
        return None


_bootstrap()
ss = st.session_state
ss.setdefault("current_task_id", None)
ss.setdefault("progress", {})
ss.setdefault("approval_shown_at", {})
ss.setdefault("approval_mode", {})
ss.setdefault("require_plan_approval", settings.require_plan_approval)
ss.setdefault("require_action_approval", settings.require_action_approval)


# =========================================================================================
# Sidebar
# =========================================================================================
with st.sidebar:
    st.title("AI Agent Orchestrator")
    st.markdown("**Model:**  \nGemini 3.1 Flash Lite")
    st.caption(f"`{settings.gemini_model}`")
    connected, conn_message = _connection_status()
    st.markdown("**Status:**  \n" + ("🟢 API Connected" if connected else "🔴 API Disconnected"))
    if not connected:
        st.caption(conn_message)
    if st.button("Re-check connection", width="stretch"):
        _connection_status.clear()
        st.rerun()

    st.divider()
    st.markdown("**Human-in-the-loop**")
    st.toggle("Approve plan before execution", key="require_plan_approval",
              help="Pause after planning: Approve Plan / Modify Plan / Reject Plan / Take Over.")
    sensitive = [t.name for t in REGISTRY.specs() if t.sensitive]
    st.toggle(f"Approve sensitive tool calls ({', '.join(sensitive)})", key="require_action_approval",
              help="Pause before each sensitive tool call: Approve / Modify / Reject / Take Over.")
    st.caption(f"Confidence: ≥{settings.auto_threshold:.0f} auto · ≥{settings.notify_threshold:.0f} notify · "
               f"≥{settings.approve_threshold:.0f} approve action · below that, escalate")

    st.divider()
    st.markdown("**Tools**")
    for spec in REGISTRY.specs():
        state = ""
        if spec.name == "web_search" and not settings.web_search_enabled:
            state = " (disabled)"
        st.caption(f"`{spec.name}`{state} → {', '.join(sorted(spec.allowed_agents))}")
    mem = _memory_count()
    st.markdown(f"**Long-term memories:** {mem if mem is not None else 'unavailable'}")
    st.markdown(f"**Pricing:** {'configured' if settings.pricing_configured else 'Cost unavailable'}")

    st.divider()
    if ss.current_task_id and st.button("Start a new task", width="stretch"):
        ss.current_task_id = None
        st.rerun()


# =========================================================================================
# Execution helpers
# =========================================================================================
def _stream(task_id: str, events, title: str) -> None:
    log = ss.progress.setdefault(task_id, [])
    state = "complete"
    with st.status(title, expanded=True) as status:
        for event in events:
            if event.kind == "interrupt":
                icon, state = "⏸", "awaiting"
            elif event.kind == "error":
                icon, state = "✗", "error"
                st.error(event.message)
            elif event.kind == "done":
                continue
            else:
                icon = "✓" if event.ok else "⚠"
            status.write(f"{icon} {event.message}")
            log.append(f"{icon} {event.message}")
        labels = {"complete": ("Run complete", "complete"), "awaiting": ("Waiting for human review", "complete"),
                  "error": ("Run stopped with an error", "error")}
        label, status_state = labels[state]
        status.update(label=label, state=status_state, expanded=False)


def _submit_decision(task_id: str, request: ApprovalRequest, decision: str, instruction: str = "",
                     feedback: str = "") -> None:
    shown = ss.approval_shown_at.get(request.request_id, time.time())
    payload = {"decision": decision, "modified_instruction": instruction, "feedback": feedback,
               "review_seconds": round(time.time() - shown, 1)}
    ss.approval_mode.pop(request.request_id, None)
    _stream(task_id, workflow.resume_run(task_id, payload), "Resuming agent system…")
    st.rerun()


def render_human_review(task_id: str, request: ApprovalRequest) -> None:
    st.subheader("Human Review")
    ss.approval_shown_at.setdefault(request.request_id, time.time())
    level_label = {"approve_plan": "Approve Plan", "approve_action": "Approve Action", "take_over": "Take Over",
                   "notify": "Notify"}[request.level]
    st.markdown(f"<div class='hitl-banner'><b>Human approval is required before this action can continue.</b>"
                f"<br>Approval level: {level_label} · checkpoint: {request.checkpoint.replace('_', ' ')}</div>",
                unsafe_allow_html=True)
    with st.container(border=True):
        top = st.columns([2, 1, 1])
        top[0].markdown(f"**Task**  \n{request.task}")
        top[1].markdown(f"**Current Agent**  \n{AGENT_LABELS.get(request.agent, request.agent)}")
        top[2].markdown("**Confidence**  \n" + (f"{request.confidence:.0f}/100" if request.confidence is not None
                                               else "not scored yet"))
        st.markdown(f"**Proposed Action**  \n{request.proposed_action}")
        st.markdown(f"**Reason**  \n{request.reason}")
        if request.tool:
            st.markdown(f"**Tool**  \n`{request.tool}`")
            st.json(request.tool_input or {})
        st.markdown("**Relevant Memory**")
        if request.relevant_memories:
            for m in request.relevant_memories:
                st.markdown(f"- `{m.memory_type}` {m.content}")
        else:
            st.caption("No long-term memories were used for this decision.")
        if request.details:
            label = {"plan": "Proposed plan", "final_output": "Draft output", "failure": "Failed subtask",
                     "action": "Context"}[request.checkpoint]
            with st.expander(label, expanded=request.checkpoint in ("plan", "final_output")):
                st.markdown(request.details)

    feedback = st.text_input("Feedback (optional)", key=f"fb_{request.request_id}")
    is_plan = request.checkpoint == "plan"
    labels = ["Approve Plan", "Modify Plan", "Reject Plan", "Take Over"] if is_plan else \
        ["Approve", "Modify", "Reject", "Take Over"]
    cols = st.columns(4)
    keys = ["approve", "modify", "reject", "take_over"]
    for col, label, key in zip(cols, labels, keys):
        with col:
            clicked = st.button(label, key=f"{key}_{request.request_id}", width="stretch",
                                type="primary" if key == "approve" else "secondary")
            st.caption(request.options_help.get(key, ""))
        if clicked:
            if key in ("approve", "reject"):
                _submit_decision(task_id, request, key, feedback=feedback)
            else:
                ss.approval_mode[request.request_id] = key
                st.rerun()

    mode = ss.approval_mode.get(request.request_id)
    if mode:
        with st.form(key=f"form_{request.request_id}"):
            if mode == "modify":
                text = st.text_area("Human instructions", height=140,
                                    placeholder="Describe the change the agents should make…")
            else:
                placeholder = {"action": "Paste the tool result the agent should use…",
                               "failure": "Write this subtask's output…"}.get(request.checkpoint,
                                                                              "Write the final answer…")
                text = st.text_area("Your output", height=220, placeholder=placeholder)
            submit_col, cancel_col = st.columns(2)
            submitted = submit_col.form_submit_button("Submit", type="primary", width="stretch")
            cancelled = cancel_col.form_submit_button("Cancel", width="stretch")
        if cancelled:
            ss.approval_mode.pop(request.request_id, None)
            st.rerun()
        if submitted:
            if not text.strip():
                st.warning("Please enter text before submitting.")
            else:
                _submit_decision(task_id, request, mode, instruction=text.strip(), feedback=feedback)


def render_run(snap: RunSnapshot, task_id: str) -> None:
    st.markdown(f"**Status:** {views.TASK_STATUS.get(snap.task.status, snap.task.status)} · "
                f"`{snap.task.task_id}` · trace `{snap.task.trace_id}`")

    st.subheader("Execution")
    log = ss.progress.get(task_id)
    with st.container(border=True, height=260):
        if log:
            for line in log:
                st.markdown(line)
        else:
            views.render_progress_from_trace(snap)

    request = workflow.pending_approval(task_id)
    if request is not None:
        render_human_review(task_id, request)

    views.render_notifications(snap)

    st.subheader("Agent Plan")
    views.render_plan(snap)

    st.subheader("Final Answer")
    views.render_final(snap)

    with st.expander("Confidence breakdown", expanded=snap.confidence is not None and snap.final is None):
        views.render_confidence(snap)

    st.subheader("Agent Activity")
    views.render_agent_activity(snap)

    st.subheader("Tool Usage")
    views.render_tool_usage(snap)


# =========================================================================================
# Main tabs
# =========================================================================================
tab_run, tab_trace, tab_memory, tab_replay, tab_analytics = st.tabs(
    ["🧭 Workspace", "🔍 Execution Trace", "🧠 Memory", "⏪ Replay", "📊 Analytics"]
)

with tab_run:
    st.subheader("New Task")
    with st.form("new_task", clear_on_submit=False):
        request_text = st.text_area("Task", placeholder="Describe your task...", height=140,
                                    label_visibility="collapsed")
        uploads = st.file_uploader("Relevant files (optional)", type=[s.lstrip(".") for s in SUPPORTED_TYPES],
                                   accept_multiple_files=True)
        busy = bool(ss.current_task_id and workflow.pending_approval(ss.current_task_id))
        run_clicked = st.form_submit_button("Run Agent System", type="primary", disabled=not connected or busy)
    if not connected:
        st.error(f"Gemini API connection failed. Please verify GEMINI_API_KEY. ({conn_message})")
    elif busy:
        st.info("The current task is waiting for human review below. Resolve it (or start a new task) first.")

    if run_clicked:
        if len(request_text.strip()) < 5:
            st.warning("Please describe the task in a bit more detail.")
        else:
            task_id = new_id("task")
            saved, rejected = save_uploads(task_id, [(f.name, f.getvalue()) for f in uploads or []])
            for msg in rejected:
                st.warning(f"File skipped: {msg}")
            options = RunOptions(require_plan_approval=ss.require_plan_approval,
                                 require_action_approval=ss.require_action_approval)
            ss.current_task_id = task_id
            ss.progress[task_id] = []
            _stream(task_id, workflow.start_run(task_id, request_text, saved, options), "Running agent system…")
            st.rerun()

    if ss.current_task_id:
        snapshot = workflow.current_snapshot(ss.current_task_id) or load_snapshot(ss.current_task_id)
        if snapshot:
            st.divider()
            render_run(snapshot, ss.current_task_id)
    else:
        st.caption("Enter a task and run the system. Agents, tools, memory, review and human approvals will "
                   "appear here as the workflow executes.")

with tab_trace:
    if ss.current_task_id and (snap := workflow.current_snapshot(ss.current_task_id)
                               or load_snapshot(ss.current_task_id)):
        views.render_trace(snap, key="live")
    else:
        st.caption("Run a task to see its execution trace. Past traces are available under Replay.")

with tab_memory:
    current_tab, long_tab = st.tabs(["Current Memory", "Long-Term Memory"])
    with current_tab:
        if ss.current_task_id and (snap := workflow.current_snapshot(ss.current_task_id)
                                   or load_snapshot(ss.current_task_id)):
            views.render_memory_current(snap)
        else:
            st.caption("Short-term memory exists only while a task is active.")
    with long_tab:
        if not settings.memory_enabled:
            st.info("Long-term memory is disabled (MEMORY_ENABLED=false).")
        else:
            try:
                from memory.long_term import get_long_term_memory

                store = get_long_term_memory()
                records = store.list_all()
            except Exception as exc:  # noqa: BLE001
                records = []
                st.error("Long-term memory store is unavailable.")
                st.caption(str(exc)[:300])
            if records:
                types = sorted({r.memory_type for r in records})
                chosen = st.multiselect("Memory types", types, default=types)
                shown = [r for r in records if r.memory_type in chosen]
                st.dataframe([{
                    "ID": r.memory_id, "Type": r.memory_type, "Content": r.content, "Task": r.task_id,
                    "Saved": r.timestamp.strftime("%Y-%m-%d %H:%M"), "Embedding dims": r.embedding_dimensions,
                } for r in shown], hide_index=True)
                del_col, btn_col = st.columns([3, 1])
                target = del_col.selectbox("Delete a memory", [r.memory_id for r in shown], index=None,
                                           placeholder="Select a memory id…")
                if btn_col.button("Delete", disabled=target is None, width="stretch"):
                    store.delete(target)
                    st.rerun()
            elif settings.memory_enabled:
                st.caption("No long-term memories yet. They are saved after high-confidence or human-corrected runs.")

with tab_replay:
    tasks = list_tasks()
    if not tasks:
        st.caption("No saved runs yet.")
    else:
        labels = {t["task_id"]: f"{t['created_at'][:16].replace('T', ' ')} · "
                                f"{views.TASK_STATUS.get(t['status'], t['status'])} · {t['user_request'][:70]}"
                  for t in tasks}
        chosen_id = st.selectbox("Previous task", list(labels), format_func=labels.get)
        snap = load_snapshot(chosen_id) if chosen_id else None
        if snap is None:
            st.error("This run could not be loaded.")
        else:
            st.markdown("#### Original Request")
            st.markdown(snap.task.user_request)
            st.markdown("#### Original Plan")
            original = snap.plan_history[0] if snap.plan_history else []
            st.dataframe([{"Subtask": s.subtask_id, "Agent": views._label(s.assigned_agent),
                           "Dependencies": ", ".join(s.dependencies) or "—", "Description": s.description}
                          for s in original], hide_index=True)
            if len(snap.plan_history) > 1:
                st.caption(f"{len(snap.plan_history) - 1} later plan version(s); final plan below.")
                views.render_plan(snap)
            st.markdown("#### Agent Decisions")
            views.render_agent_activity(snap)
            st.markdown("#### Tool Calls")
            views.render_tool_usage(snap, key="replay")
            st.markdown("#### Memory Retrieval")
            views.render_memory_current(snap)
            st.markdown("#### Reviewer Feedback")
            for i, rv in enumerate(snap.review_history, start=1):
                st.markdown(f"**Review {i}** ({rv.score:.1f}/10, {'approved' if rv.approved else 'changes requested'}): "
                            f"{rv.feedback}")
            if not snap.review_history:
                st.caption("No review recorded.")
            st.markdown("#### Human Decisions")
            views.render_human_decisions(snap)
            st.markdown("#### Final Answer")
            views.render_final(snap)

            st.divider()
            st.markdown("#### Replay")
            steps = [e for e in snap.trace if e.event_type == "node"]
            if steps:
                children: dict[str, list] = {}
                for e in snap.trace:
                    if e.parent_id:
                        children.setdefault(e.parent_id, []).append(e)
                step = st.slider("Step", 1, len(steps), len(steps), key=f"replay_step_{chosen_id}")
                play = st.button("▶ Replay", key=f"replay_play_{chosen_id}")
                area = st.empty()

                def draw(upto: int) -> None:
                    tokens = tool_calls = 0
                    lines = []
                    for idx, node in enumerate(steps[:upto], start=1):
                        tokens += node.token_usage.total_tokens if node.token_usage else 0
                        tool_calls += sum(1 for c in children.get(node.event_id, []) if c.event_type == "tool_call")
                        icon = {"success": "✓", "error": "✗", "skipped": "⏭"}.get(node.status, "•")
                        lines.append(f"{icon} {idx}. {views._label(node.agent)} · {node.node} — "
                                     f"{views._short(node.output or '', 110)}")
                    current = steps[upto - 1]
                    with area.container(border=True):
                        c = st.columns(3)
                        c[0].metric("Step", f"{upto}/{len(steps)}")
                        c[1].metric("Tokens so far", f"{tokens:,}")
                        c[2].metric("Tool calls so far", tool_calls)
                        st.code("\n".join(lines), language="text")
                        st.markdown(f"**Current step:** {views._label(current.agent)} · `{current.node}`")
                        st.json({"node": current.model_dump(mode="json", exclude_none=True),
                                 "children": [k.model_dump(mode="json", exclude_none=True)
                                              for k in children.get(current.event_id, [])]}, expanded=False)

                if play:
                    for i in range(1, len(steps) + 1):
                        draw(i)
                        time.sleep(0.35)
                else:
                    draw(step)
            st.markdown("#### Full trace")
            views.render_trace(snap, key=f"replay_{chosen_id}")

with tab_analytics:
    stats = analytics()
    history = list_tasks()
    if not stats["total"]:
        st.caption("Analytics appear after the first run.")
    else:
        r1 = st.columns(4)
        r1[0].metric("Tasks processed", stats["total"])
        r1[1].metric("Successful tasks", stats["successful"])
        r1[2].metric("Failed tasks", stats["failed"])
        r1[3].metric("Human escalations", stats["escalations"])
        r2 = st.columns(4)
        r2[0].metric("Success rate", f"{stats['success_rate'] * 100:.0f}%")
        r2[1].metric("Human escalation rate", f"{stats['escalation_rate'] * 100:.0f}%")
        r2[2].metric("Average latency", f"{stats['avg_latency']:.1f}s")
        r2[3].metric("Average tokens", f"{stats['avg_tokens']:,.0f}")
        r3 = st.columns(4)
        r3[0].metric("Average LLM calls", f"{stats['avg_llm_calls']:.1f}")
        r3[1].metric("Average tool calls", f"{stats['avg_tool_calls']:.1f}")
        r3[2].metric("Average confidence", f"{stats['avg_confidence']:.0f}")
        r3[3].metric("Rejected by human", stats["rejected"])
        st.caption(f"Total tokens across all runs: {stats['total_tokens']:,} · Cost: "
                   + ("see per-run metrics" if settings.pricing_configured else "Cost unavailable (no pricing configured)"))
        left, right = st.columns(2)
        with left:
            st.markdown("**LLM usage by agent**")
            st.dataframe([{"Agent": views._label(r["agent"]), "LLM calls": r["llm_calls"],
                           "Avg latency (ms)": round(r["avg_latency_ms"] or 0), "Tokens": r["tokens"]}
                          for r in stats["by_agent"]], hide_index=True)
        with right:
            st.markdown("**Tool usage**")
            st.dataframe([{"Tool": r["tool"], "Calls": r["calls"], "Successes": r["successes"],
                           "Avg latency (ms)": round(r["avg_latency_ms"] or 0)} for r in stats["by_tool"]],
                         hide_index=True)
        st.markdown("**Task history**")
        st.dataframe([{"Created": t["created_at"][:16].replace("T", " "), "Status": t["status"],
                       "Request": t["user_request"][:90], "Confidence": t["confidence"], "Tokens": t["total_tokens"],
                       "LLM calls": t["llm_calls"], "Tool calls": t["tool_calls"],
                       "Time (s)": t["execution_time_s"], "Escalations": t["human_escalations"]} for t in history],
                     hide_index=True)
