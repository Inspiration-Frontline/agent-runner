from dataclasses import dataclass


@dataclass(frozen=True)
class ModelTokenDelta:
    """One text fragment emitted by the model adapter."""

    content: str


@dataclass(frozen=True)
class ModelToolStarted:
    """A model-emitted Tool Call observed before SDK execution begins."""

    tool_name: str
    tool_call_id: str
    arguments_json: str


@dataclass(frozen=True)
class ModelToolCompleted:
    """The terminal result returned by a Tool handler to the Agents SDK."""

    tool_name: str
    tool_call_id: str
    result: object
    status: str


@dataclass(frozen=True)
class ModelUsage:
    """Provider token usage for one completed model response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ModelError:
    """A model-adapter failure that can be exposed as a public stream error."""

    message: str


ModelStreamEvent = ModelTokenDelta | ModelToolStarted | ModelToolCompleted | ModelUsage | ModelError
