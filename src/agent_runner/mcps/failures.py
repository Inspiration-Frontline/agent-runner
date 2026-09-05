"""Typed, credential-safe classification for MCP boundary failures."""

from dataclasses import dataclass
from enum import StrEnum

import httpx
from agents.exceptions import UserError
from mcp.shared.exceptions import McpError

from agent_runner.mcps.secrets import McpSecretUnavailableError


class McpFailureCode(StrEnum):
    """Stable failure vocabulary shared by diagnostics, traces, and public errors."""

    SECRET_MISSING = "MCP_SECRET_MISSING"
    AUTHENTICATION_REJECTED = "MCP_AUTHENTICATION_REJECTED"
    AUTHORIZATION_DENIED = "MCP_AUTHORIZATION_DENIED"
    CONNECTION_FAILED = "MCP_CONNECTION_FAILED"
    TIMEOUT = "MCP_TIMEOUT"
    PROTOCOL_FAILED = "MCP_PROTOCOL_FAILED"
    SERVER_DISABLED = "MCP_SERVER_DISABLED"
    UNKNOWN = "MCP_UNKNOWN_FAILURE"


@dataclass(frozen=True)
class McpFailure:
    """One classified MCP failure without exception text, URLs, headers, or Secret values."""

    code: McpFailureCode
    """Stable public failure classification."""
    public_message: str
    """Client-safe message that excludes exception and credential details."""
    transport_failed: bool
    """Whether the live transport should be evicted from the pool."""
    definitely_not_delivered: bool
    """Whether retrying cannot duplicate a remote side effect."""


_FAILURES = {
    McpFailureCode.SECRET_MISSING: McpFailure(
        McpFailureCode.SECRET_MISSING,
        "MCP credentials are unavailable.",
        False,
        True,
    ),
    McpFailureCode.AUTHENTICATION_REJECTED: McpFailure(
        McpFailureCode.AUTHENTICATION_REJECTED,
        "MCP server rejected the configured credentials.",
        False,
        True,
    ),
    McpFailureCode.AUTHORIZATION_DENIED: McpFailure(
        McpFailureCode.AUTHORIZATION_DENIED,
        "MCP credentials do not permit this operation.",
        False,
        True,
    ),
    McpFailureCode.CONNECTION_FAILED: McpFailure(
        McpFailureCode.CONNECTION_FAILED,
        "MCP server could not be reached.",
        True,
        True,
    ),
    McpFailureCode.TIMEOUT: McpFailure(
        McpFailureCode.TIMEOUT,
        "MCP server request timed out.",
        True,
        False,
    ),
    McpFailureCode.PROTOCOL_FAILED: McpFailure(
        McpFailureCode.PROTOCOL_FAILED,
        "MCP server protocol negotiation failed.",
        True,
        True,
    ),
    McpFailureCode.SERVER_DISABLED: McpFailure(
        McpFailureCode.SERVER_DISABLED,
        "MCP server is disabled.",
        False,
        True,
    ),
    McpFailureCode.UNKNOWN: McpFailure(
        McpFailureCode.UNKNOWN,
        "MCP server request failed.",
        True,
        False,
    ),
}


def mcp_failure(code: McpFailureCode) -> McpFailure:
    """Return the immutable failure metadata for a known stable code.

    Args:
        code: Stable domain failure code to resolve.

    Returns:
        Immutable failure metadata for the supplied stable code.
    """

    return _FAILURES[code]


def classify_mcp_failure(error: BaseException) -> McpFailure:
    """Classify nested SDK/HTTP failures without inspecting credential-bearing messages.

    Args:
        error: Exception or failure being classified or recorded.

    Returns:
        Classified nested SDK/HTTP failures without inspecting credential-bearing messages.
    """
    errors: tuple[BaseException, ...] = tuple(_walk_exceptions(error))

    if any(isinstance(item, McpSecretUnavailableError) for item in errors):
        return mcp_failure(McpFailureCode.SECRET_MISSING)

    if any(_has_http_status(item, 401) for item in errors):
        return mcp_failure(McpFailureCode.AUTHENTICATION_REJECTED)

    if any(_has_http_status(item, 403) for item in errors):
        return mcp_failure(McpFailureCode.AUTHORIZATION_DENIED)

    if any(isinstance(item, (httpx.TimeoutException, TimeoutError)) for item in errors):
        return mcp_failure(McpFailureCode.TIMEOUT)

    if any(isinstance(item, (httpx.TransportError, ConnectionError)) for item in errors):
        return mcp_failure(McpFailureCode.CONNECTION_FAILED)

    if any(isinstance(item, (McpError, UserError)) for item in errors):
        return mcp_failure(McpFailureCode.PROTOCOL_FAILED)

    return mcp_failure(McpFailureCode.UNKNOWN)


def _walk_exceptions(error: BaseException) -> list[BaseException]:
    """Flatten exception groups and causal chains once, tolerating cyclic custom errors.

    Args:
        error: Exception or failure being classified or recorded.

    Returns:
        Unique exceptions from groups and causal chains in traversal order.
    """
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    flattened: list[BaseException] = []

    while pending:
        current: BaseException = pending.pop()

        if id(current) in visited:
            continue
        visited.add(id(current))
        flattened.append(current)

        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)

        if current.__cause__ is not None:
            pending.append(current.__cause__)

        if current.__context__ is not None:
            pending.append(current.__context__)

    return flattened


def _has_http_status(error: BaseException, expected_status: int) -> bool:
    """Read an HTTP status structurally from httpx without rendering its request URL.

    Args:
        error: Exception or failure being classified or recorded.
        expected_status: Domain expected status value used by the operation.

    Returns:
        ``True`` when the exception is an HTTP error with the expected status code.
    """

    return isinstance(error, httpx.HTTPStatusError) and error.response.status_code == expected_status
