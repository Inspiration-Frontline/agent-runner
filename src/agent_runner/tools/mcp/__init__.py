from typing import Any


class MCPToolRegistry:
    _tools: dict[str, Any] = {}

    @classmethod
    def register(cls, tool_id: str, handler: Any):
        cls._tools[tool_id] = handler

    @classmethod
    def get(cls, tool_id: str) -> Any | None:
        return cls._tools.get(tool_id)

    @classmethod
    def list_tools(cls) -> list[str]:
        return list(cls._tools.keys())
