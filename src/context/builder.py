import logging
from dataclasses import dataclass, field
from typing import Any

from config import AgentConfig, settings

from .profile_adapter import ProfileAdapter
from .prompt_assembler import PromptAssembler
from .rag_adapter import RAGAdapter
from .truncation import TokenBudgetManager

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    A single message in a conversation.

    Represents a message exchanged between the user and the agent,
    including role, content, and optional metadata.

    Attributes:
        role: The role of the message sender: 'user', 'assistant', or 'system'.
        content: The text content of the message.
        metadata: Optional metadata associated with this message.
    """

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationHistory:
    """
    Container for conversation history.

    Maintains the complete history of messages exchanged in a conversation,
    identified by a unique conversation ID.

    Attributes:
        conversation_id: Unique identifier for this conversation.
        messages: List of messages in chronological order.
    """

    conversation_id: str
    messages: list[Message] = field(default_factory=list)


@dataclass
class AgentContext:
    """
    Complete execution context for an agent request.

    Contains all information needed to execute an agent request,
    including configuration, prompts, history, profile, RAG data,
    current message, and tool specifications.

    Attributes:
        agent_config: The agent configuration for this request.
        system_prompt: The assembled system prompt with profile and RAG data.
        conversation_history: Previous messages in this conversation.
        user_profile: User profile data retrieved from profile service.
        rag_chunks: RAG chunks retrieved from knowledge service.
        current_message: The current user message being processed.
        tool_specs: Specifications of tools available to this agent.
    """

    agent_config: AgentConfig
    system_prompt: str
    conversation_history: list[Message]
    user_profile: dict[str, Any]
    rag_chunks: list[dict[str, Any]]
    current_message: Message
    tool_specs: list[dict[str, Any]]


class ContextBuilder:
    """
    Builder for constructing agent execution context.

    This class orchestrates the retrieval and assembly of all context
    components needed for agent execution, including conversation history,
    user profile, RAG chunks, and tool specifications.

    Attributes:
        profile_adapter: Adapter for retrieving user profile data.
        rag_adapter: Adapter for retrieving RAG chunks.
        prompt_assembler: Assembler for constructing the system prompt.
        token_budget_manager: Manager for token budget and truncation.
    """

    def __init__(self):
        """
        Initialize the context builder with required adapters.
        """
        self.profile_adapter = ProfileAdapter()
        self.rag_adapter = RAGAdapter()
        self.prompt_assembler = PromptAssembler()
        self.token_budget_manager = TokenBudgetManager(max_tokens=settings.max_context_tokens)

    async def build(
        self,
        agent_config: AgentConfig,
        conversation_id: str | None,
        user_id: str,
        current_message: str,
    ) -> AgentContext:
        """
        Build the complete execution context for an agent request.

        Args:
            agent_config: Configuration for the agent to execute.
            conversation_id: Optional ID of the conversation to continue.
            user_id: ID of the user making the request.
            current_message: The current user message text.

        Returns:
            AgentContext: The complete execution context.
        """
        conversation_history = await self._load_conversation_history(conversation_id)

        user_profile = {}
        if agent_config.memory_policy.profile:
            user_profile = await self.profile_adapter.retrieve(user_id)

        rag_chunks = []
        if agent_config.memory_policy.rag:
            rag_chunks = await self.rag_adapter.retrieve(
                query=current_message,
                agent_id=agent_config.agent_id,
                user_id=user_id,
            )

        system_prompt = self.prompt_assembler.assemble(
            base_prompt=agent_config.system_prompt,
            user_profile=user_profile,
            rag_chunks=rag_chunks,
        )

        tool_specs = await self._load_tool_specs(agent_config.tools)

        return AgentContext(
            agent_config=agent_config,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            user_profile=user_profile,
            rag_chunks=rag_chunks,
            current_message=Message(role="user", content=current_message),
            tool_specs=tool_specs,
        )

    async def _load_conversation_history(self, conversation_id: str | None) -> list[Message]:
        """
        Load conversation history from storage.

        Args:
            conversation_id: Optional ID of the conversation to load.

        Returns:
            list[Message]: List of messages in the conversation history.
        """
        if not conversation_id:
            return []

        return []

    async def _load_tool_specs(self, tool_ids: list[str]) -> list[dict[str, Any]]:
        """
        Load tool specifications for the agent.

        Args:
            tool_ids: List of tool IDs to load specifications for.

        Returns:
            list[dict[str, Any]]: List of tool specification dictionaries.
        """
        return []
