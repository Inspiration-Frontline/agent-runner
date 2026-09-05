import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_runner.mcps.secrets import (
    EnvironmentSecretProvider,
    McpSecretSnapshot,
    SecretProvider,
    SecretTemplateResolver,
    has_secret_reference,
    validate_secret_template,
)


class McpServerProfile(BaseModel):
    """Validated Streamable HTTP MCP server configuration for one catalog entry."""

    model_config = ConfigDict(extra="forbid")

    url: str
    """Remote Streamable HTTP endpoint, optionally containing a Secret reference."""
    # Key: outbound HTTP header name. Value: unresolved template containing a ${secret:NAME} reference.
    headers: dict[str, str] = Field(default_factory=dict)
    """Outbound headers whose values are unresolved Secret templates."""
    allow_url_secret: bool = False
    """Whether a Secret reference is allowed in the endpoint URL."""
    display_name: str = ""
    """Optional user-facing server name."""
    enabled: bool = True
    """Whether this catalog entry participates in discovery."""
    disabled_tools: frozenset[str] = Field(default_factory=frozenset)
    """Tool names excluded from the discovered server surface."""
    connection_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    """Timeout for the transport initialization handshake."""
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    """Timeout for one remote Tool request."""
    max_retry_attempts: int = Field(default=0, ge=0, le=5)
    """Maximum retries permitted after a definitely-not-delivered failure."""
    retry_backoff_seconds: float = Field(default=1.0, gt=0, le=30)
    """Delay between retry attempts."""
    schema_cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)
    """Seconds for which discovered Tool schemas remain reusable."""
    policy: str = "FULL_ACCESS"
    """MCP policy vocabulary applied to this server."""

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Reject non-HTTP transports and literal URL credentials before the server is used.

        Args:
            value: Candidate value to validate, normalize, or serialize.

        Returns:
            Validated non-HTTP transports and literal URL credentials before the server is used.
        """
        parsed: SplitResult = urlsplit(value)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("MCP Streamable HTTP URL must use http or https")

        if parsed.username or parsed.password:
            raise ValueError("MCP URL must not contain literal credentials")

        return validate_secret_template(value)

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, value: dict[str, str]) -> dict[str, str]:
        """Validate header names and require values to contain Secret references.

        Args:
            value: Candidate value to validate, normalize, or serialize.

        Returns:
            Validated header names and require values to contain Secret references.
        """
        normalized: dict[str, str] = {}
        for name, header_value in value.items():
            clean_name: str = name.strip()

            if not clean_name or any(char in clean_name for char in "\r\n:"):
                raise ValueError("MCP header name is invalid")

            if any(char in header_value for char in "\r\n"):
                raise ValueError("MCP header value is invalid")

            normalized[clean_name] = validate_secret_template(header_value, require_reference=True)

        return normalized

    @model_validator(mode="after")
    def validate_profile(self) -> "McpServerProfile":
        """Reject unsupported Policy and implicit URL credentials before connection setup.

        Returns:
            Validated unsupported Policy and implicit URL credentials before connection setup.
        """

        if self.policy != "FULL_ACCESS":
            raise ValueError("Currently only FULL_ACCESS is supported.")

        if has_secret_reference(self.url) and not self.allow_url_secret:
            raise ValueError("MCP URL Secret references require allow_url_secret=true")

        return self


class McpServerCatalogDocument(BaseModel):
    """Schema of the catalog JSON document whose root uses the standard mcpServers property."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Key: stable MCP server ID. Value: validated connection and policy profile.
    mcp_servers: dict[str, McpServerProfile] = Field(alias="mcpServers")
    """Validated profiles keyed by stable MCP Server ID."""

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def reject_process_servers(cls, value: Any) -> Any:
        """Reject process-launch profiles because this runner supports remote Streamable HTTP only.

        Args:
            value: Candidate value to validate, normalize, or serialize.

        Returns:
            Validated process-launch profiles because this runner supports remote Streamable HTTP only.
        """

        if isinstance(value, dict):
            for server_id, profile in value.items():
                if isinstance(profile, dict) and "command" in profile:
                    raise ValueError(f"MCP server {server_id} uses process transport; Streamable HTTP is required")

        return value


@dataclass(frozen=True)
class ResolvedMcpServer:
    """Catalog profile after secret templates have been resolved for one request session."""

    server_id: str
    """Stable catalog identifier of the resolved server."""
    profile: McpServerProfile
    """Validated source profile retained for timeout and policy settings."""
    url: str
    """Resolved endpoint URL used only when creating the SDK client."""
    # Key: outbound HTTP header name. Value: resolved secret header value for connection creation.
    headers: dict[str, str]
    """Resolved outbound headers used only at transport creation."""
    configuration_revision: int
    """Nacos snapshot revision from which this resolution was derived."""


class McpServerCatalog:
    """Loads validated MCP catalog configuration and resolves secrets only at the network boundary.

    Attributes:
        _profiles: Collection of profiles consumed in deterministic order.
        _secrets: Provider of immutable Secret snapshots used during Catalog resolution.
    """

    def __init__(self, document: McpServerCatalogDocument, secrets: SecretProvider | None = None) -> None:
        """Create an in-memory catalog from a validated document and an optional secret provider.

        Args:
            document: Validated Catalog document containing server profiles.
            secrets: Provider of immutable Secret snapshots used during Catalog resolution.
        """
        self._profiles = dict(document.mcp_servers)
        self._secrets = secrets or EnvironmentSecretProvider()

    @classmethod
    def from_file(cls, path: Path, secrets: SecretProvider | None = None) -> "McpServerCatalog":
        """Read, validate, and retain a catalog JSON file without resolving secrets yet.

        Args:
            path: JSON file containing the validated Catalog document.
            secrets: Provider of immutable Secret snapshots used during Catalog resolution.

        Returns:
            Catalog loaded from the file, with Secret references unresolved.
        """

        payload: Any = json.loads(path.read_text(encoding="utf-8-sig"))
        return cls(McpServerCatalogDocument.model_validate(payload), secrets)

    @classmethod
    def from_json(cls, payload: str, secrets: SecretProvider | None = None) -> "McpServerCatalog":
        """Read, validate, and retain catalog JSON supplied by an in-memory configuration source.

        Args:
            payload: Serialized external payload to parse and validate.
            secrets: Provider of immutable Secret snapshots used during Catalog resolution.

        Returns:
            Catalog loaded from the JSON payload, with Secret references unresolved.
        """

        return cls(McpServerCatalogDocument.model_validate_json(payload), secrets)

    @classmethod
    def empty(cls) -> "McpServerCatalog":
        """Return a catalog that intentionally declares no external MCP servers.

        Returns:
            Empty catalog with no external MCP server declarations.
        """

        return cls(McpServerCatalogDocument.model_validate({"mcpServers": {}}))

    def contains(self, server_id: str) -> bool:
        """Return whether the catalog declares the requested server ID.

        Args:
            server_id: Stable MCP Catalog server identifier.

        Returns:
            ``True`` when the catalog declares the requested server ID.
        """

        return server_id in self._profiles

    def resolve(self, server_id: str) -> ResolvedMcpServer:
        """Resolve one enabled server's secret templates immediately before SDK connection setup.

        Args:
            server_id: Stable MCP Catalog server identifier.

        Returns:
            resolve one enabled server's secret templates immediately before SDK connection setup.
        """
        profile: McpServerProfile | None = self._profiles.get(server_id)

        if profile is None:
            raise ValueError(f"Unknown MCP server: {server_id}")

        snapshot: McpSecretSnapshot = self._secrets.get_snapshot()
        template_resolver: SecretTemplateResolver = SecretTemplateResolver(snapshot)
        url: str = template_resolver.resolve(profile.url)
        headers: dict[str, str] = {name: template_resolver.resolve(value) for name, value in profile.headers.items()}

        return ResolvedMcpServer(server_id, profile, url, headers, snapshot.configuration_revision)
