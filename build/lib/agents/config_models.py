from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryPolicy:
    profile: bool = True
    rag: bool = True


@dataclass
class AgentDefinition:
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


@dataclass
class ToolSpec:
    tool_id: str
    name: str
    description: str
    parameters: dict[str, Any]
    tool_type: str = "internal"


@dataclass
class MCPServerSpec:
    server_id: str
    name: str
    transport: str
    config: dict[str, Any]
