"""The single place where the Gemini API is called.

Uses the current `google-genai` SDK. Provides:
  * generate_structured() - JSON output constrained to a Pydantic schema, validated locally
  * generate_text()       - free-form text
  * grounded_search()     - Google Search grounding (backs the web_search tool)
  * embed()               - embeddings for long-term memory
  * check_connection()    - cheap connectivity/auth check for the sidebar status

All calls share one client, one model setting, one retry policy and one token accounting path.
"""

from __future__ import annotations

import copy
import json
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from core.config import get_logger, get_settings
from core.schemas import LLMCallInfo, TokenUsage

log = get_logger("llm")
T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None
_client_lock = threading.Lock()


class LLMError(Exception):
    """Raised when Gemini cannot produce a usable response. `user_message` is UI-safe."""

    def __init__(self, user_message: str, *, detail: str = "", retryable: bool = False):
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.retryable = retryable


class LLMConfigError(LLMError):
    pass


def get_client() -> genai.Client:
    global _client
    settings = get_settings()
    if not settings.api_key_configured:
        raise LLMConfigError("Gemini API connection failed. Please verify GEMINI_API_KEY.", detail="GEMINI_API_KEY missing")
    with _client_lock:
        if _client is None:
            _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _usage(response: Any) -> TokenUsage:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return TokenUsage()
    prompt = meta.prompt_token_count or 0
    output = (meta.candidates_token_count or 0) + (getattr(meta, "thoughts_token_count", 0) or 0)
    total = meta.total_token_count or (prompt + output)
    return TokenUsage(input_tokens=prompt, output_tokens=output, total_tokens=total)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve `$ref`/`$defs` and drop titles so the schema is as portable as possible."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


def _config(system: str | None, *, schema: type[BaseModel] | None = None, tools: list | None = None,
            temperature: float | None = None) -> types.GenerateContentConfig:
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "system_instruction": system,
        "temperature": settings.temperature if temperature is None else temperature,
    }
    if schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_json_schema"] = _inline_refs(schema.model_json_schema())
    if tools:
        kwargs["tools"] = tools
    if settings.thinking_level:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=settings.thinking_level)
    return types.GenerateContentConfig(**kwargs)


def _classify(exc: Exception) -> LLMError:
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, genai_errors.ClientError):
        code = getattr(exc, "code", None)
        if code in (401, 403):
            return LLMConfigError("Gemini API connection failed. Please verify GEMINI_API_KEY.", detail=str(exc))
        if code == 404:
            return LLMConfigError(
                f"Gemini model '{get_settings().gemini_model}' is not available for this API key.", detail=str(exc)
            )
        if code == 429:
            return LLMError("Gemini rate limit reached; retrying.", detail=str(exc), retryable=True)
        return LLMError("Gemini rejected the request.", detail=str(exc))
    if isinstance(exc, genai_errors.ServerError):
        return LLMError("Gemini service error; retrying.", detail=str(exc), retryable=True)
    # Network / timeout errors from the HTTP layer.
    return LLMError("Could not reach the Gemini API.", detail=f"{type(exc).__name__}: {exc}", retryable=True)


def _generate(contents: Any, config: types.GenerateContentConfig) -> tuple[Any, float, int]:
    """Call Gemini with exponential backoff on transient failures. Returns (response, latency_ms, attempts)."""
    settings = get_settings()
    client = get_client()
    last: LLMError | None = None
    for attempt in range(1, settings.llm_max_retries + 1):
        started = time.perf_counter()
        try:
            response = client.models.generate_content(model=settings.gemini_model, contents=contents, config=config)
            return response, (time.perf_counter() - started) * 1000, attempt
        except Exception as exc:  # noqa: BLE001 - classified below
            last = _classify(exc)
            log.warning("Gemini call failed (attempt %s/%s): %s", attempt, settings.llm_max_retries, last)
            if not last.retryable or attempt == settings.llm_max_retries:
                break
            time.sleep(settings.llm_retry_base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5))
    assert last is not None
    raise last


def _text(response: Any) -> str:
    try:
        text = response.text
    except Exception:  # noqa: BLE001 - SDK raises on blocked/empty candidates
        text = None
    if not text:
        reason = ""
        try:
            reason = str(response.candidates[0].finish_reason)
        except Exception:  # noqa: BLE001
            pass
        raise LLMError("Gemini returned an empty response.", detail=f"finish_reason={reason}")
    return text


def generate_structured(schema: type[T], prompt: str, *, system: str, temperature: float | None = None,
                        parse_retries: int = 2) -> tuple[T, LLMCallInfo]:
    """Return a validated instance of `schema`. Invalid JSON is fed back to the model for correction."""
    settings = get_settings()
    config = _config(system, schema=schema, temperature=temperature)
    usage, latency, attempts = TokenUsage(), 0.0, 0
    current_prompt = prompt
    last_error = ""
    for _ in range(parse_retries + 1):
        response, ms, tries = _generate(current_prompt, config)
        usage, latency, attempts = usage + _usage(response), latency + ms, attempts + tries
        raw = _text(response)
        try:
            parsed = schema.model_validate_json(raw)
            return parsed, LLMCallInfo(model=settings.gemini_model, usage=usage, latency_ms=latency, attempts=attempts)
        except ValidationError as exc:
            last_error = str(exc)[:1500]
            log.warning("Structured output failed validation for %s: %s", schema.__name__, last_error)
            current_prompt = (
                f"{prompt}\n\nYour previous response did not match the required JSON schema.\n"
                f"Validation errors:\n{last_error}\nReturn corrected JSON only."
            )
    raise LLMError("Gemini returned output that did not match the expected structure.", detail=last_error)


def generate_text(prompt: str, *, system: str, temperature: float | None = None) -> tuple[str, LLMCallInfo]:
    settings = get_settings()
    response, ms, attempts = _generate(prompt, _config(system, temperature=temperature))
    return _text(response), LLMCallInfo(model=settings.gemini_model, usage=_usage(response), latency_ms=ms,
                                        attempts=attempts)


@dataclass
class GroundedSearchResult:
    text: str
    chunks: list[dict[str, str]]  # {title, url, domain}
    chunk_snippets: dict[int, list[str]]
    queries: list[str]
    info: LLMCallInfo


def grounded_search(query: str) -> GroundedSearchResult:
    """Run a Google-Search-grounded generation. Sources come only from grounding metadata."""
    settings = get_settings()
    config = _config(
        "You are a search assistant. Use Google Search to find current, factual information. "
        "Report findings concisely and neutrally.",
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.0,
    )
    response, ms, attempts = _generate(f"Search the web and summarize the most relevant findings for: {query}", config)
    text = _text(response)
    chunks: list[dict[str, str]] = []
    snippets: dict[int, list[str]] = {}
    queries: list[str] = []
    try:
        meta = response.candidates[0].grounding_metadata
    except Exception:  # noqa: BLE001
        meta = None
    if meta is not None:
        queries = list(meta.web_search_queries or [])
        for chunk in meta.grounding_chunks or []:
            web = getattr(chunk, "web", None)
            if web is not None and web.uri:
                chunks.append({"title": web.title or web.domain or "", "url": web.uri, "domain": web.domain or web.title or ""})
            else:
                chunks.append({})
        for support in meta.grounding_supports or []:
            segment_text = support.segment.text if support.segment and support.segment.text else ""
            for idx in support.grounding_chunk_indices or []:
                snippets.setdefault(idx, []).append(segment_text)
    return GroundedSearchResult(
        text=text,
        chunks=chunks,
        chunk_snippets=snippets,
        queries=queries,
        info=LLMCallInfo(model=settings.gemini_model, usage=_usage(response), latency_ms=ms, attempts=attempts),
    )


def embed(text: str, *, is_query: bool) -> list[float]:
    """Embed one text. gemini-embedding-2 takes task instructions inline in the content."""
    settings = get_settings()
    client = get_client()
    model = settings.embedding_model
    if model.startswith("gemini-embedding-2"):
        content = f"task: search result | query: {text}" if is_query else f"title: none | text: {text}"
        config = types.EmbedContentConfig(output_dimensionality=settings.embedding_dimensions)
    else:
        content = text
        config = types.EmbedContentConfig(
            output_dimensionality=settings.embedding_dimensions,
            task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
        )
    last: LLMError | None = None
    for attempt in range(1, settings.llm_max_retries + 1):
        try:
            result = client.models.embed_content(model=model, contents=content, config=config)
            return list(result.embeddings[0].values)
        except Exception as exc:  # noqa: BLE001
            last = _classify(exc)
            if not last.retryable or attempt == settings.llm_max_retries:
                break
            time.sleep(settings.llm_retry_base_delay * (2 ** (attempt - 1)))
    assert last is not None
    raise last


def check_connection() -> tuple[bool, str]:
    """Metadata lookup for the configured model; costs no tokens."""
    settings = get_settings()
    if not settings.api_key_configured:
        return False, "GEMINI_API_KEY is not set."
    try:
        get_client().models.get(model=settings.gemini_model)
        return True, "Connected"
    except Exception as exc:  # noqa: BLE001
        err = _classify(exc)
        log.error("Gemini connection check failed: %s", err)
        return False, err.user_message


def to_prompt_json(data: Any, limit: int | None = None) -> str:
    """Serialize context for prompts, truncating very long payloads."""
    text = json.dumps(data, ensure_ascii=False, indent=1, default=str)
    if limit and len(text) > limit:
        return text[:limit] + "\n...[truncated]"
    return text
