import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SECRET_PATTERN = re.compile(r"^\$\{secret:([A-Za-z_][A-Za-z0-9_]*)}$")


class SecretProvider(Protocol):
    """Resolves configuration-time secret references immediately before an outbound call."""

    def resolve(self, reference: str) -> str:
        """Return the secret value identified by one validated catalog reference."""


class EnvironmentSecretProvider:
    """Secret provider that reads approved ${secret:ENV_NAME} references from the process environment."""

    def resolve(self, reference: str) -> str:
        """Resolve one environment-variable reference without logging its value."""
        match = SECRET_PATTERN.fullmatch(reference)
        if match is None:
            raise ValueError("MCP credentials must use ${secret:ENV_NAME} references")
        value = os.getenv(match.group(1))
        if not value:
            raise ValueError(f"MCP secret is unavailable: {match.group(1)}")
        return value


class McpServerProfile(BaseModel):
    """Validated Streamable HTTP MCP server configuration for one catalog entry."""
    model_config = ConfigDict(extra="forbid")

    url: str
    # Key: outbound HTTP header name. Value: unresolved ${secret:NAME} reference.
    headers: dict[str, str] = Field(default_factory=dict)
    display_name: str = ""
    enabled: bool = True
    disabled_tools: frozenset[str] = Field(default_factory=frozenset)
    connection_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    max_retry_attempts: int = Field(default=0, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=1.0, gt=0, le=30)
    schema_cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)
    policy: str = "FULL_ACCESS"

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Reject non-HTTP transports and literal URL credentials before the server is used."""
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("MCP Streamable HTTP URL must use http or https")
        if parsed.username or parsed.password:
            raise ValueError("MCP URL must not contain literal credentials")
        for token in re.findall(r"\$\{secret:[^}]+}", value):
            if SECRET_PATTERN.fullmatch(token) is None:
                raise ValueError("MCP URL contains an invalid secret reference")
        return value

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, value: dict[str, str]) -> dict[str, str]:
        """Validate header names and require values to be secret references."""
        normalized: dict[str, str] = {}
        for name, header_value in value.items():
            clean_name = name.strip()
            if not clean_name or any(char in clean_name for char in "\r\n:"):
                raise ValueError("MCP header name is invalid")
            if SECRET_PATTERN.fullmatch(header_value) is None:
                raise ValueError("MCP header values must be secret references")
            normalized[clean_name] = header_value
        return normalized

    @model_validator(mode="after")
    def validate_policy(self) -> "McpServerProfile":
        """Keep the currently supported policy surface explicit and fail fast for unsupported values."""
        if self.policy != "FULL_ACCESS":
            raise ValueError("Currently only FULL_ACCESS is supported.")
        return self


class McpServerCatalogDocument(BaseModel):
    """Schema of the catalog JSON document whose root uses the standard mcpServers property."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Key: stable MCP server ID. Value: validated connection and policy profile.
    mcp_servers: dict[str, McpServerProfile] = Field(alias="mcpServers")

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def reject_process_servers(cls, value: Any) -> Any:
        """Reject process-launch profiles because this runner supports remote Streamable HTTP only."""
        if isinstance(value, dict):
            for server_id, profile in value.items():
                if isinstance(profile, dict) and "command" in profile:
                    raise ValueError(f"MCP server {server_id} uses process transport; Streamable HTTP is required")
        return value


@dataclass(frozen=True)
class ResolvedMcpServer:
    """Catalog profile after secret templates have been resolved for one request session."""
    server_id: str
    profile: McpServerProfile
    url: str
    # Key: outbound HTTP header name. Value: resolved secret header value for connection creation.
    headers: dict[str, str]


class McpServerCatalog:
    """Loads validated MCP catalog configuration and resolves secrets only at the network boundary."""

    def __init__(self, document: McpServerCatalogDocument, secrets: SecretProvider | None = None) -> None:
        """Create an in-memory catalog from a validated document and an optional secret provider."""
        self._profiles = dict(document.mcp_servers)
        self._secrets = secrets or EnvironmentSecretProvider()

    @classmethod
    def from_file(cls, path: Path, secrets: SecretProvider | None = None) -> "McpServerCatalog":
        """Read, validate, and retain a catalog JSON file without resolving secrets yet."""
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return cls(McpServerCatalogDocument.model_validate(payload), secrets)

    @classmethod
    def from_json(cls, payload: str, secrets: SecretProvider | None = None) -> "McpServerCatalog":
        """Read, validate, and retain catalog JSON supplied by an in-memory configuration source."""
        return cls(McpServerCatalogDocument.model_validate_json(payload), secrets)

    @classmethod
    def empty(cls) -> "McpServerCatalog":
        """Return a catalog that intentionally declares no external MCP servers."""
        return cls(McpServerCatalogDocument.model_validate({"mcpServers": {}}))

    def contains(self, server_id: str) -> bool:
        """Return whether the catalog declares the requested server ID."""
        return server_id in self._profiles

    def resolve(self, server_id: str) -> ResolvedMcpServer:
        """Resolve one enabled server's secret templates immediately before SDK connection setup."""
        profile = self._profiles.get(server_id)
        if profile is None:
            raise ValueError(f"Unknown MCP server: {server_id}")
        url = self._resolve_template(profile.url)
        headers = {name: self._secrets.resolve(value) for name, value in profile.headers.items()}
        return ResolvedMcpServer(server_id, profile, url, headers)

    def _resolve_template(self, value: str) -> str:
        """Replace each validated ${secret:...} token while retaining non-secret URL text."""
        return re.sub(r"\$\{secret:[^}]+}", lambda match: self._secrets.resolve(match.group(0)), value)
