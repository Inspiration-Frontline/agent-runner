import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

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

    Contains the tool's identity, interface, handler function,
    and type classification for registry management.

    Attributes:
        tool_key: Globally unique and permanently stable Tool identity.
        tool_name: Provider-facing function name exposed to the LLM.
        description: Detailed description of tool's functionality.
        parameters: Parameter schema defining tool's input structure.
        strict: Whether strict JSON Schema argument generation is requested.
        definition_hash: SHA-256 digest of the canonical normalized definition.
        handler: Optional handler function for tool execution.
        source_type: Origin from which the Tool is resolved and executed.
    """

    tool_key: str
    tool_name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = False
    definition_hash: str = field(init=False)
    handler: Callable | None = None
    source_type: ToolSourceType = ToolSourceType.INTERNAL

    def __post_init__(self):
        """Calculate the initial audit digest of this normalized definition."""
        canonical_definition = {
            "description": self.description,
            "parameters": self.parameters,
            "source_type": self.source_type.value,
            "strict": self.strict,
            "tool_key": self.tool_key,
            "tool_name": self.tool_name,
        }
        canonical_json = json.dumps(
            canonical_definition,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.definition_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class ToolRegistry:
    """
    Registry for managing available tools.

    Provides registration, retrieval, and categorization of tools,
    supporting Tool lookup by stable key, source, and specification generation.

    Attributes:
        _tools: Dictionary mapping Tool keys to Tool definitions.
        _tools_by_source: Dictionary mapping source types to lists of Tool keys.
    """

    def __init__(self):
        """
        Initialize the tool registry with empty collections.
        """
        self._tools: dict[str, ToolDefinition] = {}
        self._tools_by_source: dict[ToolSourceType, list[str]] = {
            ToolSourceType.INTERNAL: [],
            ToolSourceType.BUSINESS: [],
            ToolSourceType.MCP: [],
        }

    def register(self, tool: ToolDefinition):
        """
        Register a tool in the registry.

        Args:
            tool: The tool definition to register.
        """
        existing = self._tools.get(tool.tool_key)
        if existing and existing.source_type in self._tools_by_source:
            self._tools_by_source[existing.source_type].remove(tool.tool_key)

        self._tools[tool.tool_key] = tool
        if tool.source_type in self._tools_by_source:
            self._tools_by_source[tool.source_type].append(tool.tool_key)
        logger.info("Registered Tool: %s (source: %s)", tool.tool_key, tool.source_type)

    def unregister(self, tool_key: str):
        """
        Unregister a tool from the registry.

        Args:
            tool_key: Globally unique key of the Tool to unregister.
        """
        if tool_key in self._tools:
            tool = self._tools[tool_key]
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
        tool_keys = self._tools_by_source.get(source_type, [])
        return [self._tools[key] for key in tool_keys if key in self._tools]

    def get_tool_specs(self, tool_keys: list[str]) -> list[dict[str, Any]]:
        """
        Generate OpenAI-compatible tool specifications.

        Args:
            tool_keys: Globally unique keys of Tools to generate specs for.

        Returns:
            list[dict[str, Any]]: List of tool specifications in OpenAI format.
        """
        specs = []
        for tool_key in tool_keys:
            tool = self.get(tool_key)
            if tool:
                specs.append({
                    "type": "function",
                    "function": {
                        "name": tool.tool_name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": tool.strict,
                    },
                })
        return specs


class BaseTool(ABC):
    """
    Abstract base class for tool implementations.

    Provides a standard interface for defining tools with
    identity, interface, and execution logic.
    """

    @property
    @abstractmethod
    def tool_key(self) -> str:
        """
        Get the globally unique and permanently stable identity for this Tool.

        Returns:
            str: The stable global Tool key.
        """
        pass

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """
        Get the provider-facing function name exposed to the LLM.

        Returns:
            str: The tool name.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get the detailed description of this tool.

        Returns:
            str: The tool description.
        """
        pass

    @property
    def parameters(self) -> dict[str, Any]:
        """
        Get the parameter schema for this tool.

        Returns:
            dict[str, Any]: Parameter schema dictionary.
        """
        return {}

    @property
    def strict(self) -> bool:
        """
        Return whether strict JSON Schema argument generation is requested.

        Returns:
            bool: False unless the Tool explicitly opts in.
        """
        return False

    @property
    def source_type(self) -> ToolSourceType:
        """
        Get the origin from which this Tool is resolved and executed.

        Returns:
            ToolSourceType: Tool origin (default: INTERNAL).
        """
        return ToolSourceType.INTERNAL

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given arguments.

        Args:
            **kwargs: Arguments for tool execution.

        Returns:
            Any: The result of tool execution.
        """
        pass

    def to_definition(self) -> ToolDefinition:
        """
        Convert this tool instance to a ToolDefinition.

        Returns:
            ToolDefinition: The tool definition for this instance.
        """
        return ToolDefinition(
            tool_key=self.tool_key,
            tool_name=self.tool_name,
            description=self.description,
            parameters=self.parameters,
            strict=self.strict,
            handler=self.execute,
            source_type=self.source_type,
        )
