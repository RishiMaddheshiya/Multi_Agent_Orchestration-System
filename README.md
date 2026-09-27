# Agent Orchestration System

A multi-agent orchestration platform with tool use, persistent memory, human-in-the-loop
approval and full execution tracing. A Supervisor breaks a user request into subtasks, specialist
agents run them (in parallel where dependencies allow) using registered tools, a Reviewer checks
the work, a transparent confidence score decides whether a human needs to step in, and the
Supervisor synthesizes the approved result. Everything is traced, persisted and replayable.

Runs locally with a single command: `streamlit run app.py`.

---

## Project Overview

| Capability | Implementation |
|---|---|
| Orchestration | LangGraph `StateGraph` with conditional edges, `Send` fan-out for parallel specialists, checkpointed `interrupt()` for human gates |
| LLM | Google Gemini `gemini-3.1-flash-lite` through the `google-genai` SDK, the only provider |
| Agents | Supervisor, Research, Analysis, Writer and Reviewer agents, each with its own role, tool permissions and output contract |
| Tools | Central registry: `web_search` (Google Search grounding), `calculator` (AST-safe), `read_file` (uploaded files only), `memory_search` |
| Memory | Short-term: the LangGraph state. Long-term: ChromaDB with Gemini embeddings, curated writes only |
| Human-in-the-loop | Notify, Approve Action, Approve Plan and Take Over, with Approve / Modify / Reject / Take Over controls |
| Confidence | Weighted blend of review score, completion, tool success, agent agreement and completeness (0-100) with configurable tiers |
| Observability | Trace ID per run; every node, LLM call, tool call, memory operation, retry, error and human decision is recorded with latency and tokens |
| Persistence | SQLite for tasks, human decisions, trace metadata and analytics; full run snapshots for replay |

## Architecture

```text
                              USER
                                |
                                v
                         STREAMLIT UI  (app.py, ui/views.py)
                                |
                                v
                     TASK INTAKE (Supervisor classifies simple / complex,
                                |  extracts preferences + memory query)
                                v
                     LONG-TERM MEMORY RETRIEVAL (ChromaDB)
                                |
                                v
                        SUPERVISOR AGENT ── plan ──> PLAN VALIDATION (deterministic guardrails)
                                                            |
                                                            v
                                               [APPROVE PLAN gate - optional]
                                                            |
                                                            v
                                                  SPECIALIST DISPATCH (dependency waves)
                                                            |
                                  +-------------------------+-------------------------+
                                  |                         |                         |
                                  v                         v                         v
                            RESEARCH AGENT            ANALYSIS AGENT             WRITER AGENT
                        (web_search, read_file,   (calculator, read_file,       (read_file)
                         memory_search, calc)      memory_search)
                                  |                         |                         |
                                  +------ tool planning -> [APPROVE ACTION gate] -> tool execution
                                                            |
                                                            v
                                                   RESULT COLLECTION
                                     (retry -> reassignment -> TAKE OVER escalation)
                                                            |
                                                            v
                                                     REVIEWER AGENT
                                                            |
                                                            v
                                                    CONFIDENCE CHECK
                             +-----------------+-----------+-----------+------------------+
                             |                 |                       |                  |
                       rejected & revisions  >= 80 auto          60-79 notify       < 60 HUMAN REVIEW
                        left: REVISE          |                       |          approve / modify /
                             |                |                       |          reject / take over
                             v                +-----------+-----------+                  |
                    back to dispatch                      v                              |
                                                   FINAL SYNTHESIS <---------------------+
                                                          |
                                                          v
                                       FINALIZE (curated memory writes, metrics, SQLite)
                                                          |
                                                          v
                                         UI: Answer | Trace | Memory | Replay | Analytics
```

### LangGraph nodes

```text
START → task_intake → memory_retrieval → supervisor → plan_validation → plan_approval
plan_approval ─ approve → specialist_dispatch │ modify → supervisor │ reject → finalize │ take over → synthesis
specialist_dispatch ─ ready wave → tool_planning ×N (parallel Send) │ nothing left → reviewer
tool_planning → action_approval (self-loop per sensitive call) → specialist_execution ×N (parallel Send)
specialist_execution → result_collection ─ failures → reassignment / failure_escalation │ else → specialist_dispatch
reviewer → confidence_check ─ revise → specialist_dispatch │ auto/notify → synthesis │ low → human_review
human_review ─ approve/take over → synthesis │ modify → specialist_dispatch │ reject → finalize
synthesis → finalize → END
```

Subtasks with no unmet dependencies are dispatched in the same wave and run concurrently through
LangGraph `Send`. Dependent subtasks wait for the wave that produces their inputs.

## Tech Stack

```text
Python 3.11+
LangGraph
Gemini API (google-genai SDK)
Gemini 3.1 Flash Lite (gemini-3.1-flash-lite), Gemini Embedding 2 (gemini-embedding-2)
Streamlit
Pydantic
ChromaDB
SQLite (Python standard library)
```

## Features

- **Multi-agent orchestration.** A LangGraph state machine with conditional routing, dependency-aware
  parallel waves, revision loops and human gates.
- **Supervisor.** Classifies complexity, decomposes and assigns work, uses or ignores retrieved
  memories, reassigns failed subtasks and synthesizes the final answer. Simple tasks skip
  decomposition and go straight to one specialist.
- **Specialist agents.** Each has its own responsibility and output contract, validated after
  generation:
  - Researcher: sourced "Key findings" notes.
  - Analyst: calculator-backed analysis with an "Implications" section.
  - Writer: the deliverable in the requested format, with no new facts.
- **Tool calling.** A registry records each tool's name, description, input/output schemas, allowed
  agents and sensitivity. Unregistered tools, disallowed tools and invalid arguments are rejected
  and logged, never executed.
- **Memory.** Short-term memory is the checkpointed graph state. Long-term memory is ChromaDB and
  stores only user preferences, human corrections, successful strategies and high-confidence task
  decisions, with near-duplicate suppression.
- **Human-in-the-loop.** Four approval levels, all of which really pause the graph (see below).
- **Confidence scoring.** Signal-based rather than the LLM's self-report, with a breakdown shown in
  the UI.
- **Execution tracing.** An agent tree, a timeline tree and a per-event explorer showing inputs,
  outputs, tool I/O, latency, tokens, model, status, errors, confidence and human decisions.
- **Replay.** Pick any saved run to see the original request, original plan, agent decisions, tool
  calls, memory retrieval, reviewer feedback, human decisions and final answer, then step through
  or play back the execution from the saved trace.
- **Analytics.** Totals, success and escalation rates, averages for latency, tokens, LLM calls and
  tool calls, plus per-agent and per-tool breakdowns.

## Project Structure

```text
agent-orchestration-system/
├── app.py                 Streamlit entry point: sidebar, workspace, human review, trace, memory, replay, analytics
├── requirements.txt
├── .env.example
├── agents/
│   ├── base.py            Shared two-phase specialist logic (tool planning → execution), citation enforcement
│   ├── supervisor.py      Intake, planning, reassignment, synthesis, citation renumbering
│   ├── researcher.py      Research Agent
│   ├── analyst.py         Analysis Agent
│   ├── writer.py          Writer Agent
│   └── reviewer.py        Reviewer Agent (structured review, never "looks good")
├── core/
│   ├── config.py          All settings (env / .env / st.secrets), logging setup
│   ├── llm.py             The only Gemini client: structured output, text, grounded search, embeddings, retries
│   ├── schemas.py         Pydantic contracts (Task, SubTask, AgentResult, ReviewResult, HumanDecision, FinalResult, …)
│   ├── confidence.py      Confidence formula and tiers
│   ├── tracing.py         Trace events and run metrics (tokens, calls, latency, cost)
│   └── storage.py         SQLite persistence and analytics queries
├── graph/
│   └── workflow.py        LangGraph nodes, routing, compilation, and the run/resume runner used by the UI
├── memory/
│   ├── short_term.py      LangGraph state definition + serializable RunSnapshot
│   └── long_term.py       ChromaDB semantic memory
├── tools/
│   ├── registry.py        Tool registry, permissions, validation, execution logging, memory_search tool
│   ├── web_search.py      Google Search grounding tool
│   ├── calculator.py      AST-restricted calculator
│   └── file_tools.py      Upload storage and read_file (.txt .md .pdf .csv)
├── ui/
│   └── views.py           Renderers shared by the live view and replay
└── data/                  Runtime data (git-ignored): orchestrator.db, chroma/, uploads/, app.log
```

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Create `.env` in the project root (see `.env.example` for every option):

```text
GEMINI_API_KEY=your_key
```

Run:

```bash
streamlit run app.py
```

For Streamlit deployment, set `GEMINI_API_KEY` in the app's secrets. It is read as
`st.secrets["GEMINI_API_KEY"]` when no environment variable is present. The key is never shown in
the UI or written to logs.

## Configuration

All settings are resolved in `core/config.py`, in this order: environment, then `.env`, then
Streamlit secrets, then defaults.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Generation model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Embeddings for long-term memory |
| `GEMINI_THINKING_LEVEL` | unset (model default) | Optional `minimal`/`low`/`medium`/`high` |
| `LLM_MAX_RETRIES` | 3 | Transport retries with exponential backoff (429/5xx/network) |
| `AGENT_MAX_RETRIES` | 2 | Specialist attempts before a subtask is marked failed |
| `MAX_REVISIONS` | 1 | Automatic reviewer-driven revision rounds |
| `CONFIDENCE_AUTO/NOTIFY/APPROVE_THRESHOLD` | 80 / 60 / 40 | Confidence tiers |
| `WEIGHT_*` | 0.30/0.20/0.20/0.15/0.15 | Confidence signal weights |
| `REQUIRE_PLAN_APPROVAL`, `REQUIRE_ACTION_APPROVAL` | false | Default state of the sidebar toggles |
| `MEMORY_TOP_K`, `MEMORY_MAX_DISTANCE`, `MEMORY_MIN_CONFIDENCE` | 5, 0.55, 70 | Retrieval and write policy |
| `WEB_SEARCH_ENABLED` | true | Disable to run without live web research |
| `GEMINI_INPUT_PRICE_PER_1M`, `GEMINI_OUTPUT_PRICE_PER_1M` | unset | Optional pricing; otherwise the UI shows "Cost unavailable" |

## Human-in-the-Loop

| Level | When it triggers | What the human can do |
|---|---|---|
| **Notify** | Confidence between the notify and auto thresholds (default 60-79); also plan-validation fixes, reassignments, memory outages | Nothing required; the run continues and the notification is shown |
| **Approve Plan** | Sidebar toggle on; after the plan is validated | Approve Plan · Modify Plan (the Supervisor re-plans with your instructions) · Reject Plan · Take Over (write the answer yourself) |
| **Approve Action** | Sidebar toggle on and an agent proposes a sensitive tool call (`web_search`), or final confidence is 40-59 | Approve · Modify (new primary argument or a JSON object of arguments, re-validated) · Reject (the call is skipped and logged as denied) · Take Over (provide the tool result or the answer) |
| **Take Over** | Final confidence below 40, or a subtask failed after retries and reassignment | Approve (skip the subtask or accept the output) · Modify (retry with instructions) · Reject (abort) · Take Over (provide the output) |

Each gate is an `interrupt()` in a node with no side effects before the interrupt, so resuming
with `Command(resume=…)` re-executes it safely. Review time is measured from when the request is
first shown and counted separately from execution time.

## Failure Recovery

1. **Transport.** Gemini calls retry with exponential backoff on 429, 5xx and network errors.
   Invalid structured output is fed back to the model with the validation errors.
2. **Agent.** A specialist retries up to `AGENT_MAX_RETRIES`, with the previous error included in
   the prompt.
3. **Reassignment.** A failed subtask is reassigned by the Supervisor (an LLM decision, with a
   rule-based fallback) to another specialist with a rewritten description.
4. **Escalation.** If the reassigned subtask also fails, a human skips it, retries it with
   instructions, aborts the task or provides the output.
5. **Degradation.** Intake, planning, review and synthesis failures fall back to labelled defaults:
   - complex classification,
   - a template plan,
   - "reviewer unavailable", which forces human review,
   - the Writer's output.

   Memory outages never block a run. Any unexpected exception is caught by the runner, logged to
   `data/app.log`, persisted as a failed run and shown to the user as a plain message.

## Architecture Decisions

- **LangGraph.** The workflow is a genuine state machine with cycles (revision, re-planning,
  escalation loops), fan-out and fan-in, and pauses. LangGraph provides conditional edges, `Send`
  for parallel dispatch, reducers for concurrent state updates and checkpointed interrupts for
  real human-in-the-loop, without writing a custom scheduler.
- **Gemini.** A single provider keeps one SDK, one auth path, one token accounting model and one
  retry policy. `gemini-3.1-flash-lite` is low-latency and inexpensive, which suits a system that
  makes 10-25 small, structured calls per task. Built-in Google Search grounding supplies a real,
  source-backed web search tool without a second API.
- **Streamlit.** The whole product is Python, so one process serves the UI and runs the graph.
  Its native widgets (status, forms, dataframes, tabs) cover progress streaming, approval forms,
  traces and analytics without a separate frontend.
- **ChromaDB.** Embedded, persistent and dependency-light semantic search. Embeddings come from
  Gemini and are passed in explicitly, so Chroma never downloads a local model and memory
  retrieval uses the same provider as the rest of the system.
- **SQLite.** Task history, decisions, trace metadata and analytics are relational and
  single-user. SQLite needs no server and ships with Python. Full run snapshots are stored as
  JSON for exact replay.
- **Human approval.** Some actions (external web queries) and some outcomes (low-confidence answers)
  shouldn't happen without oversight. The gates are opt-in for routine checkpoints (plan, action)
  and automatic for risk-driven ones (confidence, unrecoverable failure).
- **Confidence scoring.** LLM self-confidence is poorly calibrated. Combining observable signals
  (reviewer score, completion, tool success, cross-agent agreement, completeness) makes the
  escalation decision explainable, tunable and auditable, and the breakdown is shown for every
  run.

## Security

- The API key comes only from the environment, `.env` or Streamlit secrets. It is never
  hardcoded, displayed or logged (`repr=False`).
- Tools are callable only through the registry, which enforces registration, agent permissions
  and schema validation.
- The calculator walks a whitelisted AST: no `eval`, attribute access or imports, and exponents
  are bounded.
- `read_file` reads only files uploaded for the current task. Names are sanitized and resolved
  paths must stay inside `data/uploads/<task_id>/`.
- There are no shell execution, arbitrary HTTP or LLM-generated code paths. The only network
  egress is the Gemini API, including Google Search grounding.
- Citations are enforced: agents can cite only sources the `web_search` tool actually returned.
  Raw URLs and unknown source markers are stripped and flagged.

## Operational Notes

- The LangGraph checkpointer is in-memory. A run paused for human review can be resumed while the
  Streamlit server is running. If the server restarts, the paused run stays in SQLite as
  `awaiting_human` and can be inspected under Replay, but not resumed.
- Google Search grounding and embeddings are billed by Google separately from generation tokens.
  Token metrics cover generation calls, including the grounding call made by `web_search`.
- Validation is done through the application flow and the error handling described above. The
  repository contains no automated test suite.
