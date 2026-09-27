"""SQLite persistence for task history, execution metadata, human decisions, trace metadata
and analytics. Semantic memory lives in ChromaDB, not here.

Each run is stored as one `tasks` row: summary columns for analytics plus the full
RunSnapshot JSON for replay. `human_decisions` and `trace_events` are normalized copies for
querying.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

from core.config import get_logger, get_settings
from core.schemas import RunMetrics
from memory.short_term import RunSnapshot

log = get_logger("storage")
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    user_request TEXT NOT NULL,
    status TEXT NOT NULL,
    complexity TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    confidence REAL,
    human_escalations INTEGER DEFAULT 0,
    human_decisions INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    llm_calls INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    execution_time_s REAL DEFAULT 0,
    human_review_time_s REAL DEFAULT 0,
    snapshot_json TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS human_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    checkpoint TEXT, approval_level TEXT, decision TEXT, feedback TEXT,
    modified_instruction TEXT, subject TEXT, review_seconds REAL, decided_at TEXT
);
CREATE TABLE IF NOT EXISTS trace_events (
    event_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    trace_id TEXT, parent_id TEXT, timestamp TEXT, node TEXT, agent TEXT, event_type TEXT, name TEXT,
    tool TEXT, latency_ms REAL, input_tokens INTEGER, output_tokens INTEGER, status TEXT, error TEXT,
    confidence REAL, human_decision TEXT
);
CREATE INDEX IF NOT EXISTS idx_trace_task ON trace_events(task_id);
CREATE INDEX IF NOT EXISTS idx_decisions_task ON human_decisions(task_id);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(get_settings().sqlite_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.executescript(SCHEMA)


def save_run(snapshot: RunSnapshot, metrics: RunMetrics) -> None:
    task = snapshot.task
    with _lock, _connect() as conn:
        conn.execute(
            """INSERT INTO tasks (task_id, trace_id, user_request, status, complexity, created_at, completed_at,
                   confidence, human_escalations, human_decisions, input_tokens, output_tokens, total_tokens,
                   llm_calls, tool_calls, execution_time_s, human_review_time_s, snapshot_json, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
               ON CONFLICT(task_id) DO UPDATE SET status=excluded.status, complexity=excluded.complexity,
                   completed_at=excluded.completed_at, confidence=excluded.confidence,
                   human_escalations=excluded.human_escalations, human_decisions=excluded.human_decisions,
                   input_tokens=excluded.input_tokens, output_tokens=excluded.output_tokens,
                   total_tokens=excluded.total_tokens, llm_calls=excluded.llm_calls, tool_calls=excluded.tool_calls,
                   execution_time_s=excluded.execution_time_s, human_review_time_s=excluded.human_review_time_s,
                   snapshot_json=excluded.snapshot_json, updated_at=CURRENT_TIMESTAMP""",
            (
                task.task_id, task.trace_id, task.user_request, task.status, task.complexity,
                task.created_at.isoformat(), task.completed_at.isoformat() if task.completed_at else None,
                snapshot.confidence.score if snapshot.confidence else None, metrics.human_escalations,
                len(snapshot.human_decisions), metrics.input_tokens, metrics.output_tokens, metrics.total_tokens,
                metrics.llm_calls, metrics.tool_calls, metrics.execution_time_s, metrics.human_review_time_s,
                snapshot.model_dump_json(),
            ),
        )
        conn.execute("DELETE FROM human_decisions WHERE task_id = ?", (task.task_id,))
        conn.executemany(
            """INSERT INTO human_decisions (task_id, checkpoint, approval_level, decision, feedback,
                   modified_instruction, subject, review_seconds, decided_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            [(task.task_id, d.checkpoint, d.approval_level, d.decision, d.feedback, d.modified_instruction,
              d.subject, d.review_seconds, d.decided_at.isoformat()) for d in snapshot.human_decisions],
        )
        conn.execute("DELETE FROM trace_events WHERE task_id = ?", (task.task_id,))
        conn.executemany(
            """INSERT INTO trace_events (event_id, task_id, trace_id, parent_id, timestamp, node, agent, event_type,
                   name, tool, latency_ms, input_tokens, output_tokens, status, error, confidence, human_decision)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(e.event_id, task.task_id, e.trace_id, e.parent_id, e.timestamp.isoformat(), e.node, e.agent,
              e.event_type, e.name, e.tool, e.latency_ms,
              e.token_usage.input_tokens if e.token_usage else None,
              e.token_usage.output_tokens if e.token_usage else None, e.status, e.error, e.confidence,
              e.human_decision) for e in snapshot.trace],
        )
    log.info("Persisted task %s (%s)", task.task_id, task.status)


def list_tasks(limit: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT task_id, trace_id, user_request, status, created_at, confidence, total_tokens, llm_calls,
                      tool_calls, execution_time_s, human_escalations FROM tasks ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_snapshot(task_id: str) -> RunSnapshot | None:
    with _connect() as conn:
        row = conn.execute("SELECT snapshot_json FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    if not row:
        return None
    try:
        return RunSnapshot.model_validate(json.loads(row["snapshot_json"]))
    except Exception as exc:  # noqa: BLE001 - corrupted or incompatible snapshot
        log.error("Could not load snapshot %s: %s", task_id, exc)
        return None


def analytics() -> dict:
    with _connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(status = 'completed') AS successful,
                      SUM(status = 'failed') AS failed,
                      SUM(status = 'rejected') AS rejected,
                      SUM(status = 'awaiting_human') AS awaiting,
                      SUM(human_escalations > 0) AS escalated_tasks,
                      SUM(human_escalations) AS escalations,
                      AVG(CASE WHEN status IN ('completed','failed','rejected') THEN execution_time_s END) AS avg_latency,
                      AVG(total_tokens) AS avg_tokens,
                      AVG(llm_calls) AS avg_llm_calls,
                      AVG(tool_calls) AS avg_tool_calls,
                      AVG(confidence) AS avg_confidence,
                      SUM(total_tokens) AS total_tokens,
                      AVG(human_review_time_s) AS avg_human_review_s
               FROM tasks"""
        ).fetchone()
        by_agent = conn.execute(
            """SELECT agent, COUNT(*) AS llm_calls, AVG(latency_ms) AS avg_latency_ms,
                      SUM(COALESCE(input_tokens,0) + COALESCE(output_tokens,0)) AS tokens
               FROM trace_events WHERE event_type = 'llm_call' GROUP BY agent ORDER BY tokens DESC"""
        ).fetchall()
        by_tool = conn.execute(
            """SELECT tool, COUNT(*) AS calls, SUM(status = 'success') AS successes, AVG(latency_ms) AS avg_latency_ms
               FROM trace_events WHERE event_type = 'tool_call' GROUP BY tool ORDER BY calls DESC"""
        ).fetchall()
    stats = {k: (row[k] or 0) for k in row.keys()}
    finished = (stats["successful"] or 0) + (stats["failed"] or 0) + (stats["rejected"] or 0)
    stats["success_rate"] = (stats["successful"] / finished) if finished else 0.0
    stats["escalation_rate"] = (stats["escalated_tasks"] / stats["total"]) if stats["total"] else 0.0
    stats["by_agent"] = [dict(r) for r in by_agent]
    stats["by_tool"] = [dict(r) for r in by_tool]
    return stats
