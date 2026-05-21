import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    tool_id: str
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable | None = None
    tool_type: str = "internal"


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._tools_by_type: dict[str, list[str]] = {
            "internal": [],
            "mcp": [],
            "business": [],
        }

    def register(self, tool: ToolDefinition):
        self._tools[tool.tool_id] = tool
        if tool.tool_type in self._tools_by_type:
            self._tools_by_type[tool.tool_type].append(tool.tool_id)
        logger.info(f"Registered tool: {tool.tool_id} (type: {tool.tool_type})")

    def unregister(self, tool_id: str):
        if tool_id in self._tools:
            tool = self._tools[tool_id]
            if tool.tool_type in self._tools_by_type:
                self._tools_by_type[tool.tool_type].remove(tool_id)
            del self._tools[tool_id]
            logger.info(f"Unregistered tool: {tool_id}")

    def get(self, tool_id: str) -> ToolDefinition | None:
        return self._tools.get(tool_id)

    def get_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_by_type(self, tool_type: str) -> list[ToolDefinition]:
        tool_ids = self._tools_by_type.get(tool_type, [])
        return [self._tools[tid] for tid in tool_ids if tid in self._tools]

    def get_tool_specs(self, tool_ids: list[str]) -> list[dict[str, Any]]:
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
    @property
    @abstractmethod
    def tool_id(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    def parameters(self) -> dict[str, Any]:
        return {}

    @property
    def tool_type(self) -> str:
        return "internal"

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        pass

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            tool_id=self.tool_id,
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            handler=self.execute,
            tool_type=self.tool_type,
        )
