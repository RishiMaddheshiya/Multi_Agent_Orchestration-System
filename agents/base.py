"""Shared machinery for specialist agents (researcher, analyst, writer).

A specialist runs in two phases, and each phase is a separate LangGraph node, so a human
can approve sensitive tool calls between them:

  1. plan_tools(): the agent chooses calls from *its own* registered tools (structured output).
     Unknown tools, tools it isn't allowed to use, and invalid arguments are rejected here.
  2. execute():    after approved tools have run, the agent produces its deliverable from the
     tool results, dependency outputs, memories and any feedback. Failures are retried with
     the previous error included in the prompt.

Each subclass defines its own role, instructions and output contract. Citations are
checked: the agent can only cite sources that tools actually retrieved, and anything else
is removed from the output.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field

from core.config import get_settings
from core.llm import LLMConfigError, LLMError, generate_structured, to_prompt_json
from core.schemas import (
    AgentResult,
    MemoryRecord,
    PlannedToolCall,
    Source,
    SpecialistOutput,
    SubTask,
    SubtaskToolPlan,
    TokenUsage,
    ToolCallRecord,
    ToolPlan,
    UploadedFile,
    new_id,
)
from core.tracing import NodeTrace
from tools.registry import REGISTRY

URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
MARKER_RE = re.compile(r"\[(S[0-9a-f]{6})\]")


def source_id_for(url: str) -> str:
    return "S" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:6]


def sources_from_tool_calls(calls: list[ToolCallRecord]) -> list[Source]:
    """Only successful web_search results become citable sources."""
    found: dict[str, Source] = {}
    for call in calls:
        if call.tool != "web_search" or call.status != "success" or not call.output:
            continue
        for item in call.output.get("results", []):
            url = item.get("url")
            if url and url not in found:
                found[url] = Source(source_id=source_id_for(url), title=item.get("title", ""), url=url,
                                    snippet=item.get("snippet", ""), source=item.get("source", ""))
    return list(found.values())


@dataclass
class AgentContext:
    trace_id: str
    task_id: str
    user_request: str
    user_preferences: list[str] = field(default_factory=list)
    dependency_results: list[tuple[SubTask, AgentResult | None]] = field(default_factory=list)
    memories: list[MemoryRecord] = field(default_factory=list)
    uploads: list[UploadedFile] = field(default_factory=list)


class SpecialistAgent:
    name: str = ""
    label: str = ""
    role: str = ""
    instructions: str = ""
    tool_guidance: str = ""

    # ---------------------------------------------------------------------------------
    def allowed_tools(self) -> list[str]:
        return [t.name for t in REGISTRY.for_agent(self.name)]

    def _base_context(self, subtask: SubTask, ctx: AgentContext) -> str:
        settings = get_settings()
        parts = [
            f"OVERALL USER REQUEST:\n{ctx.user_request}",
            f"YOUR SUBTASK ({subtask.subtask_id}):\n{subtask.description}",
            f"EXPECTED OUTPUT:\n{subtask.expected_output or 'A complete, well-structured result for the subtask.'}",
        ]
        if ctx.user_preferences:
            parts.append("USER PREFERENCES:\n- " + "\n- ".join(ctx.user_preferences))
        if ctx.uploads:
            parts.append("UPLOADED FILES (readable with read_file if you have it):\n"
                         + "\n".join(f"- {u.file_name} ({u.file_type}, {u.size_bytes} bytes)" for u in ctx.uploads))
        if ctx.memories:
            parts.append("RELEVANT LONG-TERM MEMORIES (use only if applicable):\n" + "\n".join(
                f"- [{m.memory_type}] {m.content}" for m in ctx.memories))
        if ctx.dependency_results:
            dep_lines = []
            for dep, result in ctx.dependency_results:
                if result is None or result.status == "failed":
                    dep_lines.append(f"### {dep.subtask_id} ({dep.assigned_agent}) - NOT AVAILABLE (failed or skipped)")
                else:
                    dep_lines.append(f"### {dep.subtask_id} ({result.agent_name}): {dep.description}\n"
                                     f"{result.output[: settings.max_context_chars]}")
            parts.append("INPUTS FROM OTHER AGENTS:\n" + "\n\n".join(dep_lines))
        if subtask.revision_feedback:
            parts.append(f"REVIEWER FEEDBACK TO ADDRESS IN THIS REVISION:\n{subtask.revision_feedback}")
        if subtask.human_instruction:
            parts.append(f"HUMAN INSTRUCTION (highest priority):\n{subtask.human_instruction}")
        return "\n\n".join(parts)

    # -- phase 1: tool planning --------------------------------------------------------------
    def plan_tools(self, subtask: SubTask, ctx: AgentContext, trace: NodeTrace,
                   require_action_approval: bool) -> tuple[SubtaskToolPlan, list[ToolCallRecord]]:
        """Returns the validated plan and records for any rejected (invalid/unregistered) requests."""
        settings = get_settings()
        allowed = self.allowed_tools()
        plan = SubtaskToolPlan(subtask_id=subtask.subtask_id, agent=self.name)
        if not allowed or settings.max_tool_calls_per_subtask == 0:
            plan.reasoning = "This agent has no registered tools; it works from provided inputs."
            return plan, []

        system = (
            f"You are the {self.label}. {self.role}\n\n"
            f"Decide which tool calls (if any) you need BEFORE writing your deliverable.\n{self.tool_guidance}\n\n"
            f"AVAILABLE TOOLS (you may use ONLY these; never invent tools):\n{REGISTRY.catalog(self.name)}\n\n"
            f"Rules: at most {settings.max_tool_calls_per_subtask} calls. arguments_json must be a JSON object "
            "matching the tool arguments. Return an empty list if the provided inputs are sufficient."
        )
        response, info = generate_structured(ToolPlan, self._base_context(subtask, ctx), system=system)
        trace.llm(f"{self.name}.plan_tools", info, input=subtask.description,
                  output=response.model_dump(), confidence=response.confidence)
        plan.reasoning = response.reasoning
        plan.confidence = max(0.0, min(1.0, response.confidence))

        rejected: list[ToolCallRecord] = []
        for request in response.tool_calls[: settings.max_tool_calls_per_subtask]:
            call_id = new_id("call")
            validated, problem = REGISTRY.validate_call(self.name, request.tool, request.arguments_json)
            if problem:
                record = ToolCallRecord(call_id=call_id, tool=request.tool, agent=self.name,
                                        subtask_id=subtask.subtask_id, input={"arguments_json": request.arguments_json},
                                        status="rejected", error=problem)
                trace.tool(record)
                rejected.append(record)
                continue
            spec = REGISTRY.get(request.tool)
            needs_approval = bool(spec and spec.sensitive and require_action_approval)
            plan.calls.append(PlannedToolCall(
                call_id=call_id, tool=request.tool, arguments=validated.model_dump(mode="json"),
                reason=request.reason, approval="pending" if needs_approval else "not_required",
            ))
        return plan, rejected

    # -- phase 2: execution ----------------------------------------------------------------
    def execute(self, subtask: SubTask, ctx: AgentContext, tool_calls: list[ToolCallRecord],
                trace: NodeTrace) -> AgentResult:
        settings = get_settings()
        started = time.perf_counter()
        usage = TokenUsage()

        known_sources = {s.source_id: s for s in sources_from_tool_calls(tool_calls)}
        for _dep, result in ctx.dependency_results:
            if result:
                for src in result.sources:
                    known_sources.setdefault(src.source_id, src)

        tool_block = []
        for call in tool_calls:
            if call.status in ("success", "human_provided"):
                tool_block.append(f"- {call.tool}({to_prompt_json(call.input)}) -> "
                                  f"{to_prompt_json(call.output, settings.max_context_chars)}")
            else:
                tool_block.append(f"- {call.tool}({to_prompt_json(call.input)}) -> {call.status.upper()}: {call.error}")
        source_block = "\n".join(f"[{s.source_id}] {s.title} ({s.source}) - {s.snippet[:300]}"
                                 for s in known_sources.values())

        system = (
            f"You are the {self.label}. {self.role}\n\n{self.instructions}\n\n"
            "Citation rules: cite retrieved sources ONLY with their markers, e.g. [S1a2b3]. Never write raw URLs "
            "and never cite a source that is not in the SOURCES list. If no sources are listed, do not claim any "
            "external sources; state that live sources were not available where relevant.\n"
            "confidence is 0-1: how well your output satisfies the expected output given the evidence available."
        )
        base_prompt = self._base_context(subtask, ctx)
        base_prompt += "\n\nTOOL RESULTS:\n" + ("\n".join(tool_block) if tool_block else "(no tools were executed)")
        base_prompt += "\n\nSOURCES:\n" + (source_block or "(none)")

        attempts = 0
        last_error = subtask.last_error
        while attempts < settings.agent_max_retries:
            attempts += 1
            prompt = base_prompt
            if last_error:
                prompt += (f"\n\nPREVIOUS ATTEMPT FAILED with: {last_error}\n"
                           "Adjust your approach (be more concise, follow the schema strictly).")
            try:
                output, info = generate_structured(SpecialistOutput, prompt, system=system)
                usage = usage + info.usage
                trace.llm(f"{self.name}.execute", info, input=subtask.description, output=output.output[:800],
                          confidence=output.confidence)
                self._validate_output(output)
                text, cited, removed = self._enforce_citations(output, known_sources)
                limitations = list(output.limitations)
                if removed:
                    limitations.append(f"Removed {removed} unverified link(s)/citation(s) not backed by tool results.")
                return AgentResult(
                    agent_name=self.name, subtask_id=subtask.subtask_id, output=text,
                    confidence=max(0.0, min(1.0, output.confidence)),
                    tools_used=sorted({c.tool for c in tool_calls if c.status in ("success", "human_provided")}),
                    latency=round(time.perf_counter() - started, 2), token_usage=usage, status="success",
                    key_points=output.key_points, limitations=limitations, sources=cited, attempts=attempts,
                )
            except (LLMError, ValueError) as exc:
                last_error = getattr(exc, "user_message", None) or str(exc)
                trace.retry(f"{self.name}.execute", attempts, str(exc))
                if isinstance(exc, LLMConfigError):
                    break  # retrying cannot fix a missing/invalid key
        return AgentResult(
            agent_name=self.name, subtask_id=subtask.subtask_id, output="", confidence=0.0,
            tools_used=sorted({c.tool for c in tool_calls if c.status == "success"}),
            latency=round(time.perf_counter() - started, 2), token_usage=usage, status="failed",
            attempts=attempts, error=last_error,
        )

    # ---------------------------------------------------------------------------------
    def _validate_output(self, output: SpecialistOutput) -> None:
        if len(output.output.strip()) < 40:
            raise ValueError("Output was empty or too short to be useful.")

    @staticmethod
    def _enforce_citations(output: SpecialistOutput, known: dict[str, Source]) -> tuple[str, list[Source], int]:
        text = output.output
        removed = 0
        known_urls = {s.url for s in known.values()}

        def strip_url(match: re.Match) -> str:
            nonlocal removed
            if match.group(0) in known_urls:
                return match.group(0)
            removed += 1
            return "[unverified link removed]"

        text = URL_RE.sub(strip_url, text)

        def check_marker(match: re.Match) -> str:
            nonlocal removed
            if match.group(1) in known:
                return match.group(0)
            removed += 1
            return ""

        text = MARKER_RE.sub(check_marker, text)
        cited_ids = set(MARKER_RE.findall(text)) | {sid for sid in output.cited_source_ids if sid in known}
        cited = [known[sid] for sid in known if sid in cited_ids]
        return text, cited, removed
