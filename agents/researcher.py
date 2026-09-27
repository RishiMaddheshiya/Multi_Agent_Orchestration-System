"""Research Agent: gathers evidence with tools and returns sourced research notes."""

from agents.base import SpecialistAgent
from core.schemas import SpecialistOutput


class ResearchAgent(SpecialistAgent):
    name = "researcher"
    label = "Research Agent"
    role = ("You gather factual information for the team. You never analyze beyond what evidence supports "
            "and you never write the final deliverable.")
    tool_guidance = (
        "Use web_search for current or external facts (1-3 focused queries, not one broad query). "
        "Use read_file when the user uploaded relevant files. Use memory_search when prior project context, "
        "decisions or preferences could matter. Use calculator only for unit conversions you must report."
    )
    instructions = (
        "Produce structured research notes in Markdown with these sections:\n"
        "## Key findings - bullet list; every externally sourced fact ends with its source marker.\n"
        "## Data points - figures with units, dates and source markers (omit if none).\n"
        "## Conflicts & gaps - contradictory evidence and what could not be verified.\n"
        "Rules: distinguish sourced facts from general background knowledge (label the latter "
        "'(background, unsourced)'). Prefer recent information and say how recent it is. "
        "Do not speculate or give recommendations."
    )

    def _validate_output(self, output: SpecialistOutput) -> None:
        super()._validate_output(output)
        if "key findings" not in output.output.lower():
            raise ValueError("Research notes must include a 'Key findings' section.")
