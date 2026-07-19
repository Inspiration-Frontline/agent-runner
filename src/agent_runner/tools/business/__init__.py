from typing import Any


class BusinessToolRegistry:
    _tools: dict[str, Any] = {}

    @classmethod
    def register(cls, tool_key: str, handler: Any):
        """Register a business Tool adapter under a stable key."""
        cls._tools[tool_key] = handler

    @classmethod
    def get(cls, tool_key: str) -> Any | None:
        """Return a registered business Tool adapter, if available."""
        return cls._tools.get(tool_key)

    @classmethod
    def list_tools(cls) -> list[str]:
        """Return all registered business Tool keys."""
        return list(cls._tools.keys())
