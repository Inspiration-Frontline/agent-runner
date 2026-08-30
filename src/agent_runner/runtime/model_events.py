from dataclasses import dataclass


@dataclass(frozen=True)
class ModelTokenDelta:
    """One text fragment emitted by the model adapter."""

    content: str
    """Visible text fragment emitted by the provider stream."""


@dataclass(frozen=True)
class ModelToolStarted:
    """A model-emitted Tool Call observed before SDK execution begins."""

    tool_name: str
    """Tool name selected by the model."""
    tool_call_id: str
    """Provider-generated Tool Call identifier."""
    arguments_json: str
    """Exact serialized Tool arguments emitted by the model."""


@dataclass(frozen=True)
class ModelToolCompleted:
    """The terminal result returned by a Tool handler to the Agents SDK."""

    tool_name: str
    """Tool name whose execution completed."""
    tool_call_id: str
    """Provider-generated Tool Call identifier."""
    result: object
    """Provider-neutral Tool result value."""
    status: str
    """Terminal Tool execution status."""


@dataclass(frozen=True)
class ModelUsage:
    """Provider token usage for one completed model response."""

    prompt_tokens: int
    """Prompt token count reported by the provider."""
    completion_tokens: int
    """Completion token count reported by the provider."""
    total_tokens: int
    """Total token count reported by the provider."""


@dataclass(frozen=True)
class ModelError:
    """A model-adapter failure that can be exposed as a public stream error."""

    message: str
    """Client-safe model failure message."""


ModelStreamEvent = ModelTokenDelta | ModelToolStarted | ModelToolCompleted | ModelUsage | ModelError
