import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

SECRET_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SECRET_REFERENCE_PATTERN = re.compile(r"\$\{secret:([A-Za-z_][A-Za-z0-9_]*)}")


class McpSecretUnavailableError(ValueError):
    """Raised when a referenced MCP Secret has no non-empty configured value."""


def validate_secret_template(value: str, require_reference: bool = False) -> str:
    """Validate every Secret reference in a template and return the unchanged value.

    Args:
        value: Header or URL template that may contain ``${secret:NAME}`` references.
        require_reference: Whether a template without a Secret reference must be rejected.

    Returns:
        The original template after validation.

    Raises:
        ValueError: The template has no required reference or contains malformed Secret syntax.
    """
    references = tuple(SECRET_REFERENCE_PATTERN.finditer(value))
    if require_reference and not references:
        raise ValueError("MCP credential templates must contain a ${secret:NAME} reference")

    remaining = SECRET_REFERENCE_PATTERN.sub("", value)
    if "${secret:" in remaining:
        raise ValueError("MCP credential template contains an invalid Secret reference")

    return value


def has_secret_reference(value: str) -> bool:
    """Return whether a validated template contains at least one Secret reference."""
    return SECRET_REFERENCE_PATTERN.search(value) is not None


@dataclass(frozen=True)
class McpSecretSnapshot:
    """Immutable MCP Secret values captured from one Nacos configuration revision.

    Nacos replaces the complete snapshot atomically. A Catalog resolution uses one snapshot for
    its URL and all headers, preventing one outbound connection from mixing two revisions.
    """

    # Key: configured Secret name. Value: raw Secret value from the same Nacos configuration revision.
    values: Mapping[str, str]
    configuration_revision: int
    use_environment_fallback: bool = True

    @classmethod
    def create(
        cls,
        values: Mapping[str, str],
        configuration_revision: int,
        use_environment_fallback: bool = True,
    ) -> "McpSecretSnapshot":
        """Copy validated values into an immutable snapshot safe for one Catalog resolution."""
        normalized: dict[str, str] = {}
        for name, value in values.items():
            if SECRET_NAME_PATTERN.fullmatch(name) is None:
                raise ValueError(f"MCP Secret name is invalid: {name}")
            if not isinstance(value, str):
                raise ValueError(f"MCP Secret value must be a string: {name}")
            normalized[name] = value

        return cls(MappingProxyType(normalized), configuration_revision, use_environment_fallback)

    def resolve(self, reference: str) -> str:
        """Resolve one exact reference without exposing its value in an error or log.

        Args:
            reference: Valid ``${secret:NAME}`` reference from an MCP Server profile.

        Returns:
            The configured non-empty Secret value.

        Raises:
            ValueError: The reference is malformed or the requested Secret is unavailable.
        """
        match = SECRET_REFERENCE_PATTERN.fullmatch(reference)
        if match is None:
            raise ValueError("MCP credentials must use ${secret:NAME} references")

        secret_name = match.group(1)
        value = self.values.get(secret_name)
        if not value and self.use_environment_fallback:
            value = os.getenv(secret_name)
        if not value:
            raise McpSecretUnavailableError(f"MCP secret is unavailable: {secret_name}")

        return value


class SecretProvider(Protocol):
    """Supplies one consistent Secret snapshot for an MCP Server resolution."""

    def get_snapshot(self) -> McpSecretSnapshot:
        """Return the latest immutable Secret snapshot available to a new connection."""


class EnvironmentSecretProvider:
    """Provide the legacy environment-only Secret boundary for standalone local execution."""

    def get_snapshot(self) -> McpSecretSnapshot:
        """Return an empty snapshot whose resolver falls back to process environment variables."""
        return McpSecretSnapshot.create({}, configuration_revision=0)


class ConfigurationSecretProvider:
    """Read the latest Nacos-backed Secret snapshot at the outbound MCP boundary."""

    def __init__(self, snapshot_supplier: Callable[[], McpSecretSnapshot]) -> None:
        """Create a provider backed by the application-owned Configuration Manager."""
        self._snapshot_supplier = snapshot_supplier

    def get_snapshot(self) -> McpSecretSnapshot:
        """Return the latest atomically published configuration snapshot."""
        return self._snapshot_supplier()


@dataclass(frozen=True)
class SecretTemplateResolver:
    """Resolve every reference in one template against one immutable Secret snapshot."""

    snapshot: McpSecretSnapshot

    def resolve(self, template: str) -> str:
        """Resolve a validated URL or Header template without retaining intermediate values."""
        validate_secret_template(template)
        return SECRET_REFERENCE_PATTERN.sub(lambda match: self.snapshot.resolve(match.group(0)), template)
