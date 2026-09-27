"""Centralized configuration.

Every tunable value in the system is resolved here, once, from (in priority order):
  1. process environment variables (a local `.env` file is loaded into the environment),
  2. Streamlit secrets (`.streamlit/secrets.toml` or the deployment secrets manager),
  3. the defaults below.

No other module reads environment variables directly. The API key is stored with
`repr=False` so it never leaks into logs, tracebacks or the UI.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env", override=False)


def _raw(name: str) -> str | None:
    value = os.getenv(name)
    if value not in (None, ""):
        return value
    try:  # Streamlit secrets are optional and unavailable outside `streamlit run`.
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return None


def _str(name: str, default: str) -> str:
    return _raw(name) or default


def _int(name: str, default: int) -> int:
    value = _raw(name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    value = _raw(name)
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _optional_float(name: str) -> float | None:
    value = _raw(name)
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _bool(name: str, default: bool) -> bool:
    value = _raw(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ConfidenceWeights:
    review_score: float = 0.30
    task_completion: float = 0.20
    tool_success: float = 0.20
    agent_agreement: float = 0.15
    information_completeness: float = 0.15

    def as_dict(self) -> dict[str, float]:
        return {
            "review_score": self.review_score,
            "task_completion": self.task_completion,
            "tool_success": self.tool_success,
            "agent_agreement": self.agent_agreement,
            "information_completeness": self.information_completeness,
        }


@dataclass(frozen=True)
class Settings:
    # --- Gemini -------------------------------------------------------------------------
    gemini_api_key: str | None = field(repr=False)
    gemini_model: str
    embedding_model: str
    embedding_dimensions: int
    temperature: float
    thinking_level: str | None
    llm_max_retries: int
    llm_retry_base_delay: float

    # --- Orchestration --------------------------------------------------------------------
    agent_max_retries: int
    max_revisions: int
    max_subtasks: int
    max_tool_calls_per_subtask: int
    max_context_chars: int
    recursion_limit: int

    # --- Confidence / human-in-the-loop ----------------------------------------------------
    auto_threshold: float  # >= : continue automatically
    notify_threshold: float  # >= : continue but notify
    approve_threshold: float  # >= : approve action; below : human escalation
    confidence_weights: ConfidenceWeights
    require_plan_approval: bool
    require_action_approval: bool

    # --- Memory ---------------------------------------------------------------------------
    memory_enabled: bool
    memory_top_k: int
    memory_max_distance: float
    memory_dedupe_distance: float
    memory_min_confidence: float

    # --- Tools ----------------------------------------------------------------------------
    web_search_enabled: bool
    web_search_max_results: int
    max_upload_mb: int
    max_file_chars: int

    # --- Cost (optional; no pricing is assumed) --------------------------------------------
    input_price_per_million: float | None
    output_price_per_million: float | None

    # --- Paths / logging ------------------------------------------------------------------
    data_dir: Path
    sqlite_path: Path
    chroma_dir: Path
    uploads_dir: Path
    log_file: Path
    log_level: str

    @property
    def api_key_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def pricing_configured(self) -> bool:
        return self.input_price_per_million is not None and self.output_price_per_million is not None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(_str("DATA_DIR", str(DATA_DIR))).resolve()
    thinking = _str("GEMINI_THINKING_LEVEL", "").strip().lower() or None
    settings = Settings(
        gemini_api_key=_raw("GEMINI_API_KEY"),
        gemini_model=_str("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        embedding_model=_str("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
        embedding_dimensions=_int("GEMINI_EMBEDDING_DIMENSIONS", 768),
        temperature=_float("GEMINI_TEMPERATURE", 0.2),
        thinking_level=thinking,
        llm_max_retries=max(1, _int("LLM_MAX_RETRIES", 3)),
        llm_retry_base_delay=_float("LLM_RETRY_BASE_DELAY", 1.5),
        agent_max_retries=max(1, _int("AGENT_MAX_RETRIES", 2)),
        max_revisions=max(0, _int("MAX_REVISIONS", 1)),
        max_subtasks=max(1, _int("MAX_SUBTASKS", 6)),
        max_tool_calls_per_subtask=max(0, _int("MAX_TOOL_CALLS_PER_SUBTASK", 3)),
        max_context_chars=_int("MAX_CONTEXT_CHARS", 6000),
        recursion_limit=_int("GRAPH_RECURSION_LIMIT", 200),
        auto_threshold=_float("CONFIDENCE_AUTO_THRESHOLD", 80),
        notify_threshold=_float("CONFIDENCE_NOTIFY_THRESHOLD", 60),
        approve_threshold=_float("CONFIDENCE_APPROVE_THRESHOLD", 40),
        confidence_weights=ConfidenceWeights(
            review_score=_float("WEIGHT_REVIEW_SCORE", 0.30),
            task_completion=_float("WEIGHT_TASK_COMPLETION", 0.20),
            tool_success=_float("WEIGHT_TOOL_SUCCESS", 0.20),
            agent_agreement=_float("WEIGHT_AGENT_AGREEMENT", 0.15),
            information_completeness=_float("WEIGHT_INFORMATION_COMPLETENESS", 0.15),
        ),
        require_plan_approval=_bool("REQUIRE_PLAN_APPROVAL", False),
        require_action_approval=_bool("REQUIRE_ACTION_APPROVAL", False),
        memory_enabled=_bool("MEMORY_ENABLED", True),
        memory_top_k=_int("MEMORY_TOP_K", 5),
        memory_max_distance=_float("MEMORY_MAX_DISTANCE", 0.55),
        memory_dedupe_distance=_float("MEMORY_DEDUPE_DISTANCE", 0.08),
        memory_min_confidence=_float("MEMORY_MIN_CONFIDENCE", 70),
        web_search_enabled=_bool("WEB_SEARCH_ENABLED", True),
        web_search_max_results=_int("WEB_SEARCH_MAX_RESULTS", 5),
        max_upload_mb=_int("MAX_UPLOAD_MB", 10),
        max_file_chars=_int("MAX_FILE_CHARS", 20000),
        input_price_per_million=_optional_float("GEMINI_INPUT_PRICE_PER_1M"),
        output_price_per_million=_optional_float("GEMINI_OUTPUT_PRICE_PER_1M"),
        data_dir=data_dir,
        sqlite_path=data_dir / "orchestrator.db",
        chroma_dir=data_dir / "chroma",
        uploads_dir=data_dir / "uploads",
        log_file=data_dir / "app.log",
        log_level=_str("LOG_LEVEL", "INFO").upper(),
    )
    for path in (settings.data_dir, settings.chroma_dir, settings.uploads_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings


_LOGGING_CONFIGURED = False


def setup_logging() -> None:
    """Technical logs go to data/app.log (and stderr); the UI only shows friendly messages."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    settings = get_settings()
    root = logging.getLogger("orchestrator")
    root.setLevel(settings.log_level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    file_handler = RotatingFileHandler(settings.log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root.propagate = False
    for noisy in ("httpx", "httpcore", "chromadb", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"orchestrator.{name}")
