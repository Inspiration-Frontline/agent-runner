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
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationHistory:
    conversation_id: str
    messages: list[Message] = field(default_factory=list)


@dataclass
class AgentContext:
    agent_config: AgentConfig
    system_prompt: str
    conversation_history: list[Message]
    user_profile: dict[str, Any]
    rag_chunks: list[dict[str, Any]]
    current_message: Message
    tool_specs: list[dict[str, Any]]


class ContextBuilder:
    def __init__(self):
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
        if not conversation_id:
            return []

        return []

    async def _load_tool_specs(self, tool_ids: list[str]) -> list[dict[str, Any]]:
        return []
