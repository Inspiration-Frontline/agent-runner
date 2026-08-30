import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agents import FunctionTool

logger = logging.getLogger(__name__)


class ToolSourceType(StrEnum):
    """Origin from which agent-runner resolves and executes a Tool."""

    INTERNAL = "INTERNAL"
    BUSINESS = "BUSINESS"
    MCP = "MCP"


@dataclass
class ToolDefinition:
    """
    Definition of a tool available for agent execution.

    Adds AgentBreaker identity and provenance to the SDK-generated Tool interface.

    Attributes:
        tool_key: Globally unique and permanently stable Tool identity.
        tool_name: Provider-facing function name exposed to the LLM.
        description: Detailed description of tool's functionality.
        parameters: Parameter schema defining tool's input structure.
        strict: Whether strict JSON Schema argument generation is requested.
        definition_hash: SHA-256 digest of the canonical normalized definition.
        function_tool: SDK FunctionTool produced by OpenAI's @function_tool decorator.
        source_type: Origin from which the Tool is resolved and executed.
    """

    tool_key: str
    tool_name: str
    description: str
    # Key: JSON Schema property name. Value: schema keyword/value describing Tool arguments.
    parameters: dict[str, Any]
    strict: bool = False
    definition_hash: str = field(init=False)
    function_tool: FunctionTool | None = None
    source_type: ToolSourceType = ToolSourceType.INTERNAL

    def __post_init__(self) -> None:
        """Calculate the initial audit digest of this normalized definition."""
        canonical_definition: dict[str, object] = {
            "description": self.description,
            "parameters": self.parameters,
            "source_type": self.source_type.value,
            "strict": self.strict,
            "tool_key": self.tool_key,
            "tool_name": self.tool_name,
        }
        canonical_json: str = json.dumps(
            canonical_definition,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.definition_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @classmethod
    def from_function_tool(
        cls,
        tool_key: str,
        function_tool: FunctionTool,
        source_type: ToolSourceType = ToolSourceType.INTERNAL,
    ) -> "ToolDefinition":
        """Add AgentBreaker identity/provenance to an SDK-generated FunctionTool definition.

        Args:
            tool_key: Globally stable AgentBreaker Tool identity.
            function_tool: SDK FunctionTool whose generated schema and callback are preserved.
            source_type: Domain source type value used by the operation.

        Returns:
            Added AgentBreaker identity/provenance to an SDK-generated FunctionTool definition.
        """
        return cls(
            tool_key=tool_key,
            tool_name=function_tool.name,
            description=function_tool.description,
            parameters=function_tool.params_json_schema,
            strict=function_tool.strict_json_schema,
            function_tool=function_tool,
            source_type=source_type,
        )


class ToolRegistry:
    """
    Registry for managing available tools.

    Provides registration, retrieval, and categorization of tools,
    supporting Tool lookup by stable key, source, and specification generation.

    Attributes:
        _tools: Dictionary mapping Tool keys to Tool definitions.
        _tools_by_source: Dictionary mapping source types to lists of Tool keys.
    """

    def __init__(self) -> None:
        """
        Initialize the tool registry with empty collections.
        """
        # Key: stable Tool registry key. Value: executable definition and its source provenance.
        self._tools: dict[str, ToolDefinition] = {}
        # Key: Tool source category. Value: registration-ordered Tool keys from that source.
        self._tools_by_source: dict[ToolSourceType, list[str]] = {
            ToolSourceType.INTERNAL: [],
            ToolSourceType.BUSINESS: [],
            ToolSourceType.MCP: [],
        }

    def register(self, tool: ToolDefinition) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: The tool definition to register.
        """
        existing: ToolDefinition | None = self._tools.get(tool.tool_key)
        if existing and existing.source_type in self._tools_by_source:
            self._tools_by_source[existing.source_type].remove(tool.tool_key)

        self._tools[tool.tool_key] = tool
        if tool.source_type in self._tools_by_source:
            self._tools_by_source[tool.source_type].append(tool.tool_key)
        logger.info("Registered Tool: %s (source: %s)", tool.tool_key, tool.source_type)

    def unregister(self, tool_key: str) -> None:
        """
        Unregister a tool from the registry.

        Args:
            tool_key: Globally unique key of the Tool to unregister.
        """
        if tool_key in self._tools:
            tool: ToolDefinition = self._tools[tool_key]
            if tool.source_type in self._tools_by_source:
                self._tools_by_source[tool.source_type].remove(tool_key)
            del self._tools[tool_key]
            logger.info("Unregistered Tool: %s", tool_key)

    def get(self, tool_key: str) -> ToolDefinition | None:
        """
        Retrieve a Tool by its stable global key.

        Args:
            tool_key: Globally unique key of the Tool to retrieve.

        Returns:
            ToolDefinition | None: The tool if found, None otherwise.
        """
        return self._tools.get(tool_key)

    def get_all(self) -> list[ToolDefinition]:
        """
        Get all registered tools.

        Returns:
            list[ToolDefinition]: List of all tool definitions.
        """
        return list(self._tools.values())

    def get_by_source(self, source_type: ToolSourceType) -> list[ToolDefinition]:
        """
        Get all Tools from a specific execution source.

        Args:
            source_type: Origin of Tools to retrieve.

        Returns:
            list[ToolDefinition]: List of tools of the specified type.
        """
        tool_keys: list[str] = self._tools_by_source.get(source_type, [])
        return [self._tools[key] for key in tool_keys if key in self._tools]

    def get_tool_specs(self, tool_keys: list[str]) -> list[dict[str, Any]]:
        """
        Generate OpenAI-compatible tool specifications.

        Args:
            tool_keys: Globally unique keys of Tools to generate specs for.

        Returns:
            list[dict[str, Any]]: List of tool specifications in OpenAI format.
        """
        specs: list[dict[str, Any]] = []
        for tool_key in tool_keys:
            tool: ToolDefinition | None = self.get(tool_key)
            if tool:
                specs.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.tool_name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                            "strict": tool.strict,
                        },
                    }
                )
        return specs
