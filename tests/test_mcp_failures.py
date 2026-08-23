import logging
from collections.abc import Awaitable, Callable

import httpx
import pytest
from agents.exceptions import UserError
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import agent_runner.mcps.sdk_runtime as sdk_runtime_module
from agent_runner.agent_definitions.config_models import MCPServerBinding
from agent_runner.config import Settings
from agent_runner.mcps.catalog import McpServerCatalog, McpServerProfile, ResolvedMcpServer
from agent_runner.mcps.connection_pool import (
    McpConnectionKey,
    McpConnectionPoolSettings,
    PooledMcpConnection,
)
from agent_runner.mcps.failures import McpFailureCode, classify_mcp_failure
from agent_runner.mcps.sdk_runtime import (
    McpConnectionDiagnostic,
    RequiredMcpServerUnavailableError,
    SdkMcpRuntime,
)
from agent_runner.mcps.secrets import McpSecretSnapshot, McpSecretUnavailableError
from agent_runner.observability.logging import ExternalMcpCredentialFilter


def status_error(status_code: int, secret: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", f"https://example.test/mcp?api_key={secret}")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("credential-bearing request failed", request=request, response=response)


def test_classifies_nested_authentication_failures_without_rendering_request() -> None:
    secret = "never-observable-auth-secret"
    try:
        raise status_error(401, secret)
    except httpx.HTTPStatusError as cause:
        error = UserError("SDK connection failed")
        error.__cause__ = ExceptionGroup("connection failures", [cause])

    failure = classify_mcp_failure(error)

    assert failure.code == McpFailureCode.AUTHENTICATION_REJECTED
    assert failure.public_message == "MCP server rejected the configured credentials."
    assert secret not in failure.public_message


def test_classifies_authorization_timeout_connection_and_protocol_failures() -> None:
    assert classify_mcp_failure(status_error(403, "hidden")).code == McpFailureCode.AUTHORIZATION_DENIED
    assert classify_mcp_failure(httpx.ReadTimeout("late")).code == McpFailureCode.TIMEOUT
    assert classify_mcp_failure(httpx.ConnectError("offline")).code == McpFailureCode.CONNECTION_FAILED
    assert classify_mcp_failure(UserError("initialize failed")).code == McpFailureCode.PROTOCOL_FAILED


def test_missing_secret_uses_a_typed_value_error() -> None:
    snapshot = McpSecretSnapshot.create({}, configuration_revision=1, use_environment_fallback=False)

    try:
        snapshot.resolve("${secret:MISSING_KEY}")
    except McpSecretUnavailableError as error:
        failure = classify_mcp_failure(error)
    else:
        raise AssertionError("Missing MCP Secret did not fail")

    assert failure.code == McpFailureCode.SECRET_MISSING


def test_preflight_trace_contains_only_stable_failure_code() -> None:
    secret = "never-observable-trace-secret"
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    diagnostic = McpConnectionDiagnostic(
        "fixture",
        True,
        False,
        McpFailureCode.AUTHENTICATION_REJECTED,
        f"unsafe {secret}",
    )

    with tracer.start_as_current_span("request"):
        SdkMcpRuntime._trace_diagnostics([diagnostic])

    event = exporter.get_finished_spans()[0].events[0]
    assert event.attributes is not None
    assert event.attributes["mcp.server.failure_code"] == "MCP_AUTHENTICATION_REJECTED"
    assert secret not in str(event.attributes)


def test_preflight_logging_does_not_include_exception_text(caplog) -> None:
    secret = "never-observable-log-secret"
    failure = classify_mcp_failure(status_error(401, secret))

    with caplog.at_level(logging.WARNING, logger="agent_runner.mcps.sdk_runtime"):
        logging.getLogger("agent_runner.mcps.sdk_runtime").warning(
            "MCP connection preflight failed",
            extra={"server_id": "fixture", "mcp_failure_code": failure.code},
        )

    assert secret not in caplog.text
    assert "credential-bearing request failed" not in caplog.text


def test_external_mcp_logs_are_blocked_before_formatting_resolved_urls() -> None:
    credential_filter = ExternalMcpCredentialFilter()
    unsafe_records = [
        logging.LogRecord(
            logger_name,
            logging.ERROR,
            __file__,
            1,
            "failed URL https://example.test/mcp?api_key=never-observable",
            (),
            None,
        )
        for logger_name in ("agents.mcp.manager", "httpx", "mcp.client.streamable_http", "openai.agents")
    ]
    safe = logging.LogRecord(
        "agent_runner.mcps.sdk_runtime",
        logging.WARNING,
        __file__,
        1,
        "MCP connection preflight failed",
        (),
        None,
    )

    assert all(credential_filter.filter(record) is False for record in unsafe_records)
    assert credential_filter.filter(safe) is True


class RejectingPool:
    """Connection pool double that preserves a credential-bearing SDK exception chain."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def borrow(
        self,
        key: McpConnectionKey,
        settings: McpConnectionPoolSettings,
        creator: Callable[[], Awaitable[PooledMcpConnection]],
    ) -> PooledMcpConnection:
        raise self._error

    async def release(self, connection: PooledMcpConnection, invalidate: bool = False) -> None:
        raise AssertionError("No connection should be released after a failed borrow")


def rejecting_runtime(secret: str, status_code: int) -> SdkMcpRuntime:
    catalog = McpServerCatalog.from_json(
        '{"mcpServers":{"fixture":{"url":"https://example.test/mcp"}}}'
    )
    wrapped = UserError("SDK failed")
    wrapped.__cause__ = ExceptionGroup("connect", [status_error(status_code, secret)])
    return SdkMcpRuntime(
        catalog,
        connection_pool=RejectingPool(wrapped),  # type: ignore[arg-type]
        settings=Settings(),
    )


async def test_required_preflight_failure_is_typed_and_redacted(caplog: pytest.LogCaptureFixture) -> None:
    secret = "never-observable-required-secret"
    runtime = rejecting_runtime(secret, 401)

    with (
        caplog.at_level(logging.WARNING, logger="agent_runner.mcps.sdk_runtime"),
        pytest.raises(RequiredMcpServerUnavailableError) as raised,
    ):
        async with runtime.session([MCPServerBinding("fixture", required=True)]):
            raise AssertionError("Required failed server must not yield a session")

    assert raised.value.error_code == "MCP_REQUIRED_SERVER_UNAVAILABLE"
    assert "MCP server rejected the configured credentials." in str(raised.value)
    assert secret not in str(raised.value)
    assert secret not in caplog.text


async def test_optional_preflight_failure_degrades_with_typed_diagnostic() -> None:
    secret = "never-observable-optional-secret"
    runtime = rejecting_runtime(secret, 403)

    async with runtime.session([MCPServerBinding("fixture", required=False)]) as session:
        assert session.servers == ()
        assert session.definitions == ()
        assert len(session.diagnostics) == 1
        diagnostic = session.diagnostics[0]
        assert diagnostic.failure_code == McpFailureCode.AUTHORIZATION_DENIED
        assert diagnostic.message == "MCP credentials do not permit this operation."
        assert secret not in diagnostic.message


async def test_manager_enter_failure_is_not_cleaned_up_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave failed connection cleanup to the manager's task-affine enter state machine."""

    class FailingManager:
        instances: list["FailingManager"] = []

        def __init__(self, *args, **kwargs) -> None:
            self.cleanup_count = 0
            self.active_servers: list[object] = []
            self.instances.append(self)

        async def __aenter__(self) -> "FailingManager":
            self.cleanup_count += 1
            raise ConnectionError("fixture unavailable")

        async def cleanup_all(self) -> None:
            self.cleanup_count += 1

    monkeypatch.setattr(sdk_runtime_module, "TaskAffineMcpConnectionManager", FailingManager)
    runtime = SdkMcpRuntime(McpServerCatalog.empty(), settings=Settings())
    resolved = ResolvedMcpServer(
        "fixture",
        McpServerProfile(url="https://example.test/mcp"),
        "https://example.test/mcp",
        {},
        1,
    )
    key = McpConnectionKey.from_resolved(resolved)

    with pytest.raises(ConnectionError, match="fixture unavailable"):
        await runtime._create_connection(key, resolved)

    assert FailingManager.instances[0].cleanup_count == 1
