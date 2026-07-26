class InternalToolRegistry:
    _tools: dict[str, object] = {}

    @classmethod
    def register(cls, tool_key: str, handler: object) -> None:
        """Register a handler under a stable internal Tool key."""
        cls._tools[tool_key] = handler

    @classmethod
    def get(cls, tool_key: str) -> object | None:
        """Return the handler registered for a Tool key, if present."""
        return cls._tools.get(tool_key)

    @classmethod
    def list_tools(cls) -> list[str]:
        """Return all registered internal Tool keys."""
        return list(cls._tools.keys())
