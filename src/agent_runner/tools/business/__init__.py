from typing import Any


class BusinessToolRegistry:
    _tools: dict[str, Any] = {}

    @classmethod
    def register(cls, tool_key: str, handler: Any):
        cls._tools[tool_key] = handler

    @classmethod
    def get(cls, tool_key: str) -> Any | None:
        return cls._tools.get(tool_key)

    @classmethod
    def list_tools(cls) -> list[str]:
        return list(cls._tools.keys())
