"""Centralized tool registry.

Every tool declares a name, description, input schema, output schema, allowed agents and
whether it is sensitive (needs human approval when action approval is on). Agents can only
execute tools through `ToolRegistry.execute`, which enforces registration and permissions,
validates arguments, times the call, logs it and returns a `ToolCallRecord`. A tool that is
not registered cannot be called, whatever an LLM asks for.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from core.config import get_logger, get_settings
from core.schemas import LLMCallInfo, MemoryRecord, TokenUsage, ToolCallRecord, UploadedFile
from tools import calculator, file_tools, web_search

log = get_logger("tools")


@dataclass
class ToolContext:
    task_id: str
    uploads: list[UploadedFile] = field(default_factory=list)
    llm_calls: list[LLMCallInfo] = field(default_factory=list)

    def record_llm_call(self, info: LLMCallInfo) -> None:
        self.llm_calls.append(info)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    allowed_agents: frozenset[str]
    func: Callable[[Any, ToolContext], BaseModel]
    sensitive: bool = False
    primary_arg: str = ""  # argument replaced when a human "modifies" a proposed call

    def catalog_entry(self) -> str:
        props = self.input_model.model_json_schema().get("properties", {})
        args = ", ".join(f"{k}: {v.get('description', v.get('type', ''))}" for k, v in props.items())
        return f"- {self.name}: {self.description} Arguments: {{{args}}}"


# -- memory_search tool (lives here because it wraps the memory layer) ----------------------
class MemorySearchInput(BaseModel):
    query: str = Field(description="What to look for in long-term memory")
    top_k: int = Field(default=3, description="Maximum memories to return (1-5)")


class MemorySearchOutput(BaseModel):
    query: str
    memories: list[MemoryRecord]


def _memory_search(args: MemorySearchInput, _ctx: ToolContext) -> MemorySearchOutput:
    from memory.long_term import get_long_term_memory

    if not get_settings().memory_enabled:
        raise RuntimeError("Long-term memory is disabled (MEMORY_ENABLED=false).")
    records = get_long_term_memory().search(args.query, top_k=max(1, min(args.top_k, 5)))
    return MemorySearchOutput(query=args.query, memories=records)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def for_agent(self, agent: str) -> list[ToolSpec]:
        return [t for t in self._tools.values() if agent in t.allowed_agents]

    def catalog(self, agent: str | None = None) -> str:
        specs = self.for_agent(agent) if agent else self.specs()
        return "\n".join(t.catalog_entry() for t in specs) or "(no tools available)"

    def validate_call(self, agent: str, tool: str, arguments: dict[str, Any] | str) -> tuple[BaseModel | None, str | None]:
        """Returns (validated input, None) or (None, reason)."""
        spec = self._tools.get(tool)
        if spec is None:
            return None, f"Tool '{tool}' is not registered. Registered tools: {', '.join(self._tools)}"
        if agent not in spec.allowed_agents:
            return None, f"Agent '{agent}' is not permitted to use '{tool}'"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                return None, f"Arguments are not valid JSON: {exc.msg}"
        if not isinstance(arguments, dict):
            return None, "Arguments must be a JSON object"
        try:
            return spec.input_model.model_validate(arguments), None
        except ValidationError as exc:
            return None, f"Invalid arguments: {exc.errors()[0].get('msg', str(exc))}"

    def execute(self, *, call_id: str, agent: str, subtask_id: str, tool: str, arguments: dict[str, Any],
                context: ToolContext, approved_by_human: bool | None = None) -> ToolCallRecord:
        validated, problem = self.validate_call(agent, tool, arguments)
        if problem:
            log.warning("Rejected tool call %s by %s: %s", tool, agent, problem)
            return ToolCallRecord(call_id=call_id, tool=tool, agent=agent, subtask_id=subtask_id, input=arguments,
                                  status="rejected", error=problem, approved_by_human=approved_by_human)
        spec = self._tools[tool]
        context.llm_calls = []
        started = time.perf_counter()
        try:
            output = spec.func(validated, context)
            status, error, payload = "success", None, output.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 - tool failures are data, not crashes
            user_msg = getattr(exc, "user_message", None) or str(exc)
            status, error, payload = "error", user_msg[:1000], None
            log.warning("Tool %s failed for %s: %s", tool, agent, exc)
        latency = (time.perf_counter() - started) * 1000
        usage = TokenUsage()
        for info in context.llm_calls:
            usage = usage + info.usage
        log.info("Tool %s by %s [%s] %.0f ms", tool, agent, status, latency)
        return ToolCallRecord(
            call_id=call_id, tool=tool, agent=agent, subtask_id=subtask_id,
            input=validated.model_dump(mode="json"), output=payload, status=status, error=error,
            latency_ms=latency, token_usage=usage, llm_calls=len(context.llm_calls),
            approved_by_human=approved_by_human,
        )


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="web_search",
        description="Search the live web (Google Search grounding) and return titled, linked sources with snippets.",
        input_model=web_search.WebSearchInput, output_model=web_search.WebSearchOutput,
        allowed_agents=frozenset({"researcher"}), func=web_search.run, sensitive=True, primary_arg="query",
    ))
    registry.register(ToolSpec(
        name="calculator",
        description="Safely evaluate arithmetic, percentages, growth rates and basic statistics.",
        input_model=calculator.CalculatorInput, output_model=calculator.CalculatorOutput,
        allowed_agents=frozenset({"analyst", "researcher"}), func=calculator.run, primary_arg="expression",
    ))
    registry.register(ToolSpec(
        name="read_file",
        description="Read text and metadata from a file the user uploaded for this task (.txt, .md, .pdf, .csv).",
        input_model=file_tools.ReadFileInput, output_model=file_tools.ReadFileOutput,
        allowed_agents=frozenset({"researcher", "analyst", "writer"}), func=file_tools.run, primary_arg="file_name",
    ))
    registry.register(ToolSpec(
        name="memory_search",
        description="Semantic search over long-term memory (past decisions, preferences, corrections, strategies).",
        input_model=MemorySearchInput, output_model=MemorySearchOutput,
        allowed_agents=frozenset({"researcher", "analyst"}), func=_memory_search, primary_arg="query",
    ))
    return registry


REGISTRY = build_registry()
