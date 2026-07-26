from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileAttribute:
    """One user-profile value normalized at the HTTP adapter boundary."""

    name: str
    value: str


@dataclass(frozen=True)
class UserProfile:
    """Typed profile projection used by prompt assembly."""

    attributes: tuple[ProfileAttribute, ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "UserProfile":
        return cls(
            tuple(
                ProfileAttribute(name=str(name), value=str(value))
                for name, value in values.items()
                if value is not None and str(value)
            )
        )


@dataclass(frozen=True)
class RagChunk:
    """One knowledge retrieval result normalized from provider JSON."""

    content: str
    source: str = "Unknown"
