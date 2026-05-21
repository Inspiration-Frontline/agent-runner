import logging
from typing import Any

logger = logging.getLogger(__name__)


class PromptAssembler:
    def __init__(self):
        self.profile_section_header = "\n\n## User Profile\n\n"
        self.rag_section_header = "\n\n## Relevant Knowledge\n\n"

    def assemble(
        self,
        base_prompt: str,
        user_profile: dict[str, Any] | None = None,
        rag_chunks: list[dict[str, Any]] | None = None,
    ) -> str:
        assembled_prompt = base_prompt

        if user_profile:
            profile_section = self._format_profile(user_profile)
            if profile_section:
                assembled_prompt += self.profile_section_header + profile_section

        if rag_chunks:
            rag_section = self._format_rag_chunks(rag_chunks)
            if rag_section:
                assembled_prompt += self.rag_section_header + rag_section

        return assembled_prompt

    def _format_profile(self, profile: dict[str, Any]) -> str:
        if not profile:
            return ""

        lines = []
        for key, value in profile.items():
            if value:
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    def _format_rag_chunks(self, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return ""

        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            source = chunk.get("source", "Unknown")
            formatted_chunks.append(f"[{i}] (Source: {source})\n{content}")

        return "\n\n---\n\n".join(formatted_chunks)
