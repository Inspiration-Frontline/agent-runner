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
    rag: bool = True


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
        tools: List of tool IDs that this agent can use.
        mcp_servers: List of MCP server IDs that this agent can connect to.
        memory_policy: Memory policy defining context building behavior.
        max_output_tokens: Maximum number of tokens in agent's output.
        temperature: Temperature parameter for model sampling (0.0-2.0).
        metadata: Additional metadata for this agent definition.
        mock_response: Optional mock response for testing purposes.
    """

    agent_id: str
    version: str
    name: str
    description: str
    model: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    memory_policy: MemoryPolicy = field(default_factory=MemoryPolicy)
    max_output_tokens: int = 4096
    temperature: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)
    mock_response: str | None = None


@dataclass
class ToolSpec:
    """
    Specification of a tool available to agents.

    Defines the tool's identity, interface, and configuration
    for execution by the tool executor.

    Attributes:
        tool_id: Unique identifier for this tool.
        name: Human-readable name for this tool.
        description: Detailed description of tool's functionality.
        parameters: Parameter schema defining tool's input structure.
        tool_type: Type of tool: 'internal', 'business', or 'mcp'.
    """

    tool_id: str
    name: str
    description: str
    parameters: dict[str, Any]
    tool_type: str = "internal"


@dataclass
class MCPServerSpec:
    """
    Specification of an MCP (Model Context Protocol) server.

    Defines the server's identity, transport mechanism, and configuration
    for establishing connections and accessing tools/resources.

    Attributes:
        server_id: Unique identifier for this MCP server.
        name: Human-readable name for this MCP server.
        transport: Transport mechanism: 'stdio', 'sse', or 'websocket'.
        config: Server-specific configuration parameters.
    """

    server_id: str
    name: str
    transport: str
    config: dict[str, Any]
