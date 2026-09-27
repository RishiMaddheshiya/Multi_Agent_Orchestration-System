"""Analysis Agent: turns gathered information into computed, comparative analysis."""

from agents.base import SpecialistAgent
from core.schemas import SpecialistOutput


class AnalysisAgent(SpecialistAgent):
    name = "analyst"
    label = "Analysis Agent"
    role = ("You analyze information produced by other agents or uploaded data: you identify patterns, "
            "compare options, quantify changes and draw evidence-based implications.")
    tool_guidance = (
        "Use calculator for EVERY numeric result you will state (growth rates, percentages, averages, "
        "differences); do not do arithmetic in your head. Use read_file for uploaded data (CSV statistics). "
        "Use memory_search only if earlier analyses or decisions are relevant."
    )
    instructions = (
        "Produce structured analysis in Markdown with these sections:\n"
        "## Trends & patterns\n## Comparisons (a table when comparing 2+ items)\n"
        "## Calculations - each figure with the expression used (from calculator results)\n"
        "## Implications - what the evidence suggests, with explicit uncertainty.\n"
        "Rules: base every claim on the inputs; if inputs are insufficient, say exactly what is missing. "
        "Keep source markers from the inputs when you reuse sourced facts. Do not write the final report."
    )

    def _validate_output(self, output: SpecialistOutput) -> None:
        super()._validate_output(output)
        if "implications" not in output.output.lower():
            raise ValueError("Analysis must include an 'Implications' section.")
