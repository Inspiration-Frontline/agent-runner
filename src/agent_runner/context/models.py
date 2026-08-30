from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileAttribute:
    """One user-profile value normalized at the HTTP adapter boundary."""

    name: str
    """Profile attribute key returned by the User Profiler."""
    value: str
    """Profile attribute value supplied to prompt assembly."""


@dataclass(frozen=True)
class UserProfile:
    """Typed profile projection used by prompt assembly."""

    attributes: tuple[ProfileAttribute, ...] = ()
    """Ordered profile attributes available to the current request."""

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "UserProfile":
        """Normalize provider profile values into immutable prompt-ready attributes.

        Args:
            values: External profile mapping whose keys and non-null values become strings.

        Returns:
            A typed profile with empty and null values omitted.
        """
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
    """Retrieved knowledge text included in the assembled context."""
    source: str = "Unknown"
    """Stable source label used to identify the retrieved chunk."""
