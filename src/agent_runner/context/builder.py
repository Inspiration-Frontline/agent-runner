import logging
from dataclasses import dataclass, field
from typing import Literal, TypeIs

from agent_runner.config import AgentConfig, Settings
from agent_runner.tools.registry import ToolDefinition, ToolRegistry

from .models import RagChunk, UserProfile
from .profile_adapter import ProfileAdapter
from .prompt_assembler import PromptAssembler
from .rag_adapter import RAGAdapter
from .truncation import TokenBudgetManager

logger = logging.getLogger(__name__)

ImageDetail = Literal["low", "high", "auto", "original"]
_IMAGE_DETAILS: frozenset[ImageDetail] = frozenset({"low", "high", "auto", "original"})


def is_image_detail(value: str) -> TypeIs[ImageDetail]:
    """Return whether a string is a supported provider image detail."""
    return value in _IMAGE_DETAILS


@dataclass
class ModelTextPart:
    """Text supplied to the model as one multipart input item."""

    text: str


@dataclass
class ModelImagePart:
    """Transient image input whose signed URL exists only for the current model call."""

    file_id: str
    url: str
    detail: ImageDetail = "auto"


ModelContentPart = ModelTextPart | ModelImagePart


@dataclass
class CaptureTextPart:
    """Durable text captured for replay and persistence."""

    text: str


@dataclass
class CaptureFilePart:
    """Durable AgentBreaker file reference that can be re-signed during replay."""

    file_id: str
    detail: ImageDetail = "auto"


CaptureContentPart = CaptureTextPart | CaptureFilePart


@dataclass
class RuntimeToolCall:
    """Provider-neutral Tool Call preserved across replay and SDK adapters."""

    call_id: str
    call_type: str
    function_name: str
    arguments: str


@dataclass
class Message:
    """A strongly typed provider-neutral message.

    Scalar text remains in ``content``. Multipart model input, durable capture data, and Tool
    linkage use separate fields so an empty multipart value cannot alter scalar message semantics.

    Attributes:
        role: The role of the message sender: 'user', 'assistant', or 'system'.
        content: The text content of the message.
        model_content: Transient provider-neutral parts used for the current model call.
        capture_content: Stable parts written to persistence and used for later replay.
        tool_calls: Assistant Tool Calls emitted with this message.
        tool_call_id: Tool Call ID associated with a Tool result message.
    """

    role: str
    content: str
    model_content: tuple[ModelContentPart, ...] = ()
    capture_content: tuple[CaptureContentPart, ...] = ()
    tool_calls: tuple[RuntimeToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class CapturedMessage:
    """Provider-neutral message retained for replay before JSON/protobuf conversion."""

    role: str
    content: str
    capture_content: tuple[CaptureContentPart, ...] = ()
    tool_calls: tuple[RuntimeToolCall, ...] = ()
    tool_call_id: str | None = None


def message_to_capture(message: Message) -> CapturedMessage:
    """Convert runtime input into a typed durable capture value."""
    return CapturedMessage(
        role=message.role,
        content=message.content,
        capture_content=message.capture_content,
        tool_calls=message.tool_calls,
        tool_call_id=message.tool_call_id,
    )


def captured_message_to_dict(message: CapturedMessage) -> dict[str, object]:
    """Serialize a typed capture only at the raw JSON persistence boundary."""
    content: str | list[dict[str, object]] = message.content
    if message.capture_content:
        content = []
        for part in message.capture_content:
            if isinstance(part, CaptureTextPart):
                content.append({"type": "text", "text": part.text})
            else:
                content.append(
                    {
                        "type": "image_url",
                        "file_url": {
                            "url": f"agentbreaker-file://{part.file_id}",
                            "detail": part.detail,
                        },
                    }
                )

    captured: dict[str, object] = {"role": message.role, "content": content}
    if message.tool_calls:
        captured["tool_calls"] = [
            {
                "id": tool_call.call_id,
                "type": tool_call.call_type,
                "function": {
                    "name": tool_call.function_name,
                    "arguments": tool_call.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    if message.tool_call_id:
        captured["tool_call_id"] = message.tool_call_id
    return captured


def message_to_capture_dict(message: Message) -> dict[str, object]:
    """Compatibility serializer for callers that explicitly cross a JSON boundary."""
    return captured_message_to_dict(message_to_capture(message))


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
    """

    agent_config: AgentConfig
    system_prompt: str
    conversation_history: list[Message]
    user_profile: UserProfile
    rag_chunks: list[RagChunk]
    current_message: Message
    tool_specs: tuple[ToolDefinition, ...] = ()


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

    def __init__(self, tool_registry: ToolRegistry | None = None, settings: Settings | None = None):
        """Create adapters that assemble one request's model context.

        Args:
            tool_registry: Registry used to expose only configured Tool schemas to the agent.
        """
        current_settings = settings or Settings()
        self.profile_adapter = ProfileAdapter(current_settings)
        self.rag_adapter = RAGAdapter(current_settings)
        self.prompt_assembler = PromptAssembler()
        self.token_budget_manager = TokenBudgetManager(max_tokens=current_settings.max_context_tokens)
        self.tool_registry = tool_registry or ToolRegistry()

    async def build(
        self,
        agent_config: AgentConfig,
        conversation_id: str | None,
        user_id: int,
        current_message: Message,
        conversation_history: list[Message] | None = None,
        additional_system_instruction: str = "",
    ) -> AgentContext:
        """
        Build the complete execution context for an agent request.

        Args:
            agent_config: Configuration for the agent to execute.
            conversation_id: Optional ID of the conversation to continue.
            user_id: Trusted user ID used for profile/RAG authorization.
            current_message: Strongly typed current request and its attachment representations.
            conversation_history: Already loaded replay messages, when the orchestrator has them.
            additional_system_instruction: Internal attachment/locale instruction not shown as user text.

        Returns:
            AgentContext: Complete prompt, replay history, profile/RAG evidence, and Tool schemas.
        """
        if conversation_history is None:
            conversation_history = await self._load_conversation_history(conversation_id)

        user_profile = UserProfile()
        if agent_config.memory_policy.profile:
            user_profile = await self.profile_adapter.retrieve(user_id)

        rag_chunks: list[RagChunk] = []
        if agent_config.memory_policy.rag:
            rag_chunks = await self.rag_adapter.retrieve(
                query=current_message.content,
                agent_id=agent_config.agent_id,
                user_id=user_id,
            )

        system_prompt = self.prompt_assembler.assemble(
            base_prompt=agent_config.system_prompt,
            user_profile=user_profile,
            rag_chunks=rag_chunks,
        )
        if additional_system_instruction:
            system_prompt += "\n\n" + additional_system_instruction

        return AgentContext(
            agent_config=agent_config,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            user_profile=user_profile,
            rag_chunks=rag_chunks,
            current_message=current_message,
            tool_specs=tuple(
                definition
                for tool_key in agent_config.tools
                if (definition := self.tool_registry.get(tool_key)) is not None
            ),
        )

    @staticmethod
    async def _load_conversation_history(conversation_id: str | None) -> list[Message]:
        """
        Load conversation history from storage.

        This hook remains empty until the orchestrator's Conversation Manager replay path is used;
        keeping the boundary explicit prevents a second, inconsistent history source from appearing.

        Args:
            conversation_id: Optional ID of the conversation to load.

        Returns:
            list[Message]: Messages in chronological order, or an empty list for a new Conversation.
        """
        if not conversation_id:
            return []

        return []
