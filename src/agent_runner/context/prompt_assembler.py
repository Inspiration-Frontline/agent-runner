import logging

from agent_runner.context.models import RagChunk, UserProfile

logger = logging.getLogger(__name__)


class PromptAssembler:
    """
    Assembler for constructing agent system prompts.

    Combines the base system prompt with user profile data and RAG chunks
    to create a comprehensive system prompt for agent execution.

    Attributes:
        profile_section_header: Header for the user profile section in assembled prompts.
        rag_section_header: Header for the RAG knowledge section in assembled prompts.
    """

    def __init__(self) -> None:
        """
        Initialize the prompt assembler with section headers.
        """
        self.profile_section_header = "\n\n## User Profile\n\n"
        self.rag_section_header = "\n\n## Relevant Knowledge\n\n"

    def assemble(
        self,
        base_prompt: str,
        user_profile: UserProfile | None = None,
        rag_chunks: list[RagChunk] | None = None,
    ) -> str:
        """
        Assemble a complete system prompt from components.

        Args:
            base_prompt: The base system prompt for the agent.
            user_profile: Optional user profile data to include.
            rag_chunks: Optional RAG chunks to include.

        Returns:
            str: The assembled system prompt.
        """
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

    @staticmethod
    def _format_profile(profile: UserProfile) -> str:
        """
        Format user profile data as a markdown-style string.

        Args:
            profile: User profile dictionary to format.

        Returns:
            str: Formatted profile string, or empty string if no data.
        """
        if not profile:
            return ""

        lines = []
        for attribute in profile.attributes:
            lines.append(f"- {attribute.name}: {attribute.value}")

        return "\n".join(lines)

    @staticmethod
    def _format_rag_chunks(chunks: list[RagChunk]) -> str:
        """
        Format RAG chunks as a markdown-style string.

        Args:
            chunks: List of RAG chunk dictionaries to format.

        Returns:
            str: Formatted RAG chunks string, or empty string if no chunks.
        """
        if not chunks:
            return ""

        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            formatted_chunks.append(f"[{i}] (Source: {chunk.source})\n{chunk.content}")

        return "\n\n---\n\n".join(formatted_chunks)
