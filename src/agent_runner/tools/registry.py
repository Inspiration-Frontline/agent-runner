import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """
    Definition of a tool available for agent execution.

    Contains the tool's identity, interface, handler function,
    and type classification for registry management.

    Attributes:
        tool_id: Unique identifier for this tool.
        name: Human-readable name for this tool.
        description: Detailed description of tool's functionality.
        parameters: Parameter schema defining tool's input structure.
        handler: Optional handler function for tool execution.
        tool_type: Type of tool: 'internal', 'mcp', or 'business'.
    """

    tool_id: str
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable | None = None
    tool_type: str = "internal"


class ToolRegistry:
    """
    Registry for managing available tools.

    Provides registration, retrieval, and categorization of tools,
    supporting tool lookup by ID, type, and specification generation.

    Attributes:
        _tools: Dictionary mapping tool IDs to tool definitions.
        _tools_by_type: Dictionary mapping tool types to lists of tool IDs.
    """

    def __init__(self):
        """
        Initialize the tool registry with empty collections.
        """
        self._tools: dict[str, ToolDefinition] = {}
        self._tools_by_type: dict[str, list[str]] = {
            "internal": [],
            "mcp": [],
            "business": [],
        }

    def register(self, tool: ToolDefinition):
        """
        Register a tool in the registry.

        Args:
            tool: The tool definition to register.
        """
        self._tools[tool.tool_id] = tool
        if tool.tool_type in self._tools_by_type:
            self._tools_by_type[tool.tool_type].append(tool.tool_id)
        logger.info(f"Registered tool: {tool.tool_id} (type: {tool.tool_type})")

    def unregister(self, tool_id: str):
        """
        Unregister a tool from the registry.

        Args:
            tool_id: ID of the tool to unregister.
        """
        if tool_id in self._tools:
            tool = self._tools[tool_id]
            if tool.tool_type in self._tools_by_type:
                self._tools_by_type[tool.tool_type].remove(tool_id)
            del self._tools[tool_id]
            logger.info(f"Unregistered tool: {tool_id}")

    def get(self, tool_id: str) -> ToolDefinition | None:
        """
        Retrieve a tool by ID.

        Args:
            tool_id: ID of the tool to retrieve.

        Returns:
            ToolDefinition | None: The tool if found, None otherwise.
        """
        return self._tools.get(tool_id)

    def get_all(self) -> list[ToolDefinition]:
        """
        Get all registered tools.

        Returns:
            list[ToolDefinition]: List of all tool definitions.
        """
        return list(self._tools.values())

    def get_by_type(self, tool_type: str) -> list[ToolDefinition]:
        """
        Get all tools of a specific type.

        Args:
            tool_type: Type of tools to retrieve.

        Returns:
            list[ToolDefinition]: List of tools of the specified type.
        """
        tool_ids = self._tools_by_type.get(tool_type, [])
        return [self._tools[tid] for tid in tool_ids if tid in self._tools]

    def get_tool_specs(self, tool_ids: list[str]) -> list[dict[str, Any]]:
        """
        Generate OpenAI-compatible tool specifications.

        Args:
            tool_ids: List of tool IDs to generate specs for.

        Returns:
            list[dict[str, Any]]: List of tool specifications in OpenAI format.
        """
        specs = []
        for tool_id in tool_ids:
            tool = self.get(tool_id)
            if tool:
                specs.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
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
    def tool_id(self) -> str:
        """
        Get the unique identifier for this tool.

        Returns:
            str: The tool ID.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the human-readable name for this tool.

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
    def tool_type(self) -> str:
        """
        Get the type classification for this tool.

        Returns:
            str: Tool type (default: 'internal').
        """
        return "internal"

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
            tool_id=self.tool_id,
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            handler=self.execute,
            tool_type=self.tool_type,
        )
