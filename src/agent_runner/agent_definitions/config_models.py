from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryPolicy:
    """
    Memory policy configuration for agent context building.

    Defines which memory sources should be included when building
    the agent's execution context for each request.

    Attributes:
        profile: Whether to include user profile information in context.
        rag: Whether to include RAG (Retrieval-Augmented Generation) results in context.
    """

    profile: bool = True
    """Whether profile context is enabled for this definition."""
    rag: bool = True
    """Whether retrieval-augmented context is enabled for this definition."""

    # TODO: I don't think it is correct, I think we should:
    # 1. Load memory policy from the configuration center, it is part of the agent definition. So, I don't think we need this class, we should add "profile_keywords" & "rag_knowledge_base_ids" to the AgentDefinition class.
    # 2. "profile" & "rag" are not correct field names.
    # Because different agents may take responsibilities for different applications/businesses, they may need different memory of user profiles. Thus it should be a list[str].
    # Because an agent can access more than 1 knowledge bases, thus it should be a list[int].
    profile_keywords: list[str] = field(default_factory=list)
    """Profile attribute keywords selected by the Agent definition."""
    rag_knowledge_base_ids: list[int] = field(default_factory=list)
    """Knowledge-base identifiers selected for retrieval."""


@dataclass(frozen=True)
class MCPServerBinding:
    """One typed Agent-to-Catalog binding."""

    server_id: str
    """Stable MCP Server catalog identifier."""
    required: bool = True
    """Whether failure to connect prevents the Agent request from running."""


@dataclass
class AgentDefinition:
    """
    Complete definition of an agent instance.

    Contains all configuration needed to instantiate and execute an agent,
    including identity, model settings, tools, MCP servers, and memory policy.

    Attributes:
        agent_id: Unique identifier for this agent.
        version: Version string for this agent configuration.
        name: Human-readable name for this agent.
        description: Detailed description of agent's purpose and capabilities.
        model: Model identifier to use for this agent (e.g., 'gpt-4', 'claude-3').
        system_prompt: System prompt that defines agent's behavior and personality.
        tools: List of globally unique Tool keys that this agent can use.
        mcp_servers: List of MCP server IDs that this agent can connect to.
        memory_policy: Memory policy defining context building behavior.
        max_output_tokens: Maximum number of tokens in agent's output.
        temperature: Temperature parameter for model sampling (0.0-2.0).
        metadata: Additional metadata for this agent definition.
    """

    agent_id: int
    """Stable identifier of the configured Agent."""
    version: int
    """Version of the Agent definition."""
    name: str
    """Human-readable Agent name used in tracing and handoff."""
    description: str
    """Human-readable summary of the Agent's responsibility."""
    model: str
    """Provider model identifier used for execution."""
    system_prompt: str
    """Base system instruction supplied to the model."""
    tools: list[str] = field(default_factory=list)
    """Globally unique built-in Tool keys enabled for the Agent."""
    mcp_servers: list[MCPServerBinding] = field(default_factory=list)
    """MCP Server bindings enabled for the Agent."""
    memory_policy: MemoryPolicy = field(default_factory=MemoryPolicy)
    """External profile and retrieval context policy."""
    max_output_tokens: int = 4096
    """Maximum output tokens requested from the provider."""
    temperature: float = 0.7
    """Provider sampling temperature."""
    # Key: extension metadata name. Value: JSON-compatible Agent configuration metadata.
    metadata: dict[str, Any] = field(default_factory=dict)
