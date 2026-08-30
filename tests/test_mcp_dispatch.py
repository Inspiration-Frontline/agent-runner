import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from agent_breaker_conversation_manager_protos.ifl.agentbreaker.commons import ResponseBase
from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc import (
    AppendConversationRoundProgressRequest,
    AppendConversationRoundProgressResponse,
    ConversationRoundMutationResult,
    ToolDispatchState,
)
from agents.mcp import MCPServerStreamableHttp
from mcp.types import Tool as McpTool
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_runner.mcps.connection_pool import McpConnectionKey, PooledMcpConnection
from agent_runner.mcps.dispatch_tracing import McpTracedOperations
from agent_runner.mcps.sdk_runtime import DispatchTurnTracker, DurableMcpServer, McpSchemaCache
from agent_runner.observability.tracing import Tracer
from agent_runner.runtime.mcp_dispatch import ConversationDispatchRecorder
from agent_runner.tools.registry import ToolDefinition, ToolSourceType


class FakeRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.recovery_reasons: list[str] = []

    async def before_dispatch(
        self,
        attempt_id: str,
        tool_call_id: str,
        turn_number: int,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        assert tool_call_id == "call-7"
        assert turn_number == 2
        assert arguments == {"password": "sensitive", "value": 1}
        self.events.append(("before", attempt_id))

    async def after_dispatch(self, attempt_id: str, state: str, recovery_reason: str = "") -> None:
        self.events.append((state, attempt_id))
        self.recovery_reasons.append(recovery_reason)


class ExternalExecutionRecorder:
    def __init__(self) -> None:
        self.executions: list[tuple[str, ToolDefinition, str, str, object, str]] = []

    def record_external_execution(
        self,
        tool_call_id: str,
        definition: ToolDefinition,
        arguments_json: str,
        status: str,
        result: object,
        error_message: str,
    ) -> None:
        self.executions.append((tool_call_id, definition, arguments_json, status, result, error_message))


def pooled_transport(call_tool: Any) -> PooledMcpConnection:
    transport = MCPServerStreamableHttp.__new__(MCPServerStreamableHttp)
    transport._name = "fixture"
    transport.call_tool = call_tool
    manager = SimpleNamespace(cleanup_all=lambda: asyncio.sleep(0))
    return PooledMcpConnection(
        McpConnectionKey("fixture", "endpoint-fingerprint", "fingerprint", 1),
        transport,
        manager,
        0,
    )


async def test_durable_server_commits_intent_before_remote_call(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = FakeRecorder()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    async def call_remote(
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> object:
        assert recorder.events[0][0] == "before"
        return SimpleNamespace(isError=False)

    connection = pooled_transport(call_remote)
    server = DurableMcpServer(
        connection,
        recorder=recorder,
        schema_cache=McpSchemaCache(),
        schema_cache_ttl_seconds=60,
        tracker=DispatchTurnTracker(),
        traced_operations=McpTracedOperations(Tracer(provider=provider)),
    )

    await server.call_tool(
        "write",
        {"password": "sensitive", "value": 1},
        {"agentbreaker/tool_call_id": "call-7", "agentbreaker/turn_number": 2},
    )

    assert [event[0] for event in recorder.events] == ["before", "COMPLETED"]
    assert recorder.events[0][1] == recorder.events[1][1]
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {"mcp.dispatch", "mcp.result"}
    assert spans["mcp.result"].attributes["gen_ai.tool.call.id"] == "call-7"
    assert spans["mcp.result"].attributes["mcp.result.state"] == "COMPLETED"


async def test_durable_server_records_unknown_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = FakeRecorder()
    calls = 0

    async def call_remote(
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> object:
        nonlocal calls
        calls += 1
        raise TimeoutError("response path lost")

    connection = pooled_transport(call_remote)
    server = DurableMcpServer(
        connection,
        recorder=recorder,
        schema_cache=McpSchemaCache(),
        schema_cache_ttl_seconds=60,
        tracker=DispatchTurnTracker(),
        traced_operations=McpTracedOperations(Tracer()),
    )

    with pytest.raises(TimeoutError):
        await server.call_tool(
            "write",
            {"password": "sensitive", "value": 1},
            {"agentbreaker/tool_call_id": "call-7", "agentbreaker/turn_number": 2},
        )

    assert calls == 1
    assert [event[0] for event in recorder.events] == ["before", "UNKNOWN"]


async def test_durable_server_records_cancelled_dispatch_before_propagating_cancel() -> None:
    recorder = FakeRecorder()
    remote_started = asyncio.Event()
    release_remote = asyncio.Event()

    async def call_remote(
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> object:
        remote_started.set()
        await release_remote.wait()
        return SimpleNamespace(isError=False)

    server = DurableMcpServer(
        pooled_transport(call_remote),
        recorder=recorder,
        schema_cache=McpSchemaCache(),
        schema_cache_ttl_seconds=60,
        tracker=DispatchTurnTracker(),
        traced_operations=McpTracedOperations(Tracer()),
    )
    task = asyncio.create_task(
        server.call_tool(
            "write",
            {"password": "sensitive", "value": 1},
            {"agentbreaker/tool_call_id": "call-7", "agentbreaker/turn_number": 2},
        )
    )
    await remote_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert [event[0] for event in recorder.events] == ["before", "CANCELLED"]


async def test_durable_server_records_connect_failure_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = FakeRecorder()

    async def call_remote(
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> object:
        raise httpx.ConnectError("DNS lookup failed", request=httpx.Request("POST", "https://example.test/mcp"))

    connection = pooled_transport(call_remote)
    server = DurableMcpServer(
        connection,
        recorder=recorder,
        schema_cache=McpSchemaCache(),
        schema_cache_ttl_seconds=60,
        tracker=DispatchTurnTracker(),
        traced_operations=McpTracedOperations(Tracer()),
    )

    with pytest.raises(httpx.ConnectError):
        await server.call_tool(
            "write",
            {"password": "sensitive", "value": 1},
            {"agentbreaker/tool_call_id": "call-7", "agentbreaker/turn_number": 2},
        )

    assert [event[0] for event in recorder.events] == ["before", "FAILED"]


async def test_durable_server_redacts_credential_bearing_http_failure() -> None:
    recorder = FakeRecorder()
    secret = "never-observable-dispatch-secret"

    async def call_remote(
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> object:
        request = httpx.Request("POST", f"https://example.test/mcp?api_key={secret}")
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError("rejected", request=request, response=response)

    server = DurableMcpServer(
        pooled_transport(call_remote),
        recorder=recorder,
        schema_cache=McpSchemaCache(),
        schema_cache_ttl_seconds=60,
        tracker=DispatchTurnTracker(),
        traced_operations=McpTracedOperations(Tracer()),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await server.call_tool(
            "write",
            {"password": "sensitive", "value": 1},
            {"agentbreaker/tool_call_id": "call-7", "agentbreaker/turn_number": 2},
        )

    assert recorder.recovery_reasons == ["MCP server rejected the configured credentials."]
    assert secret not in recorder.recovery_reasons[0]


async def test_schema_cache_refresh_is_single_flight() -> None:
    cache = McpSchemaCache()
    calls = 0

    async def load() -> list[McpTool]:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return [McpTool(name="echo", inputSchema={"type": "object"})]

    results = await asyncio.gather(*(cache.get("fixture", 60, load) for _ in range(5)))

    assert calls == 1
    assert [[tool.name for tool in result] for result in results] == [["echo"]] * 5


async def test_durable_server_reuses_request_schema_when_cross_request_cache_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def call_remote(
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> object:
        return SimpleNamespace(isError=False)

    async def list_remote(
        run_context: Any = None,
        agent: Any = None,
    ) -> list[McpTool]:
        nonlocal calls
        calls += 1
        return [McpTool(name="echo", inputSchema={"type": "object"})]

    connection = pooled_transport(call_remote)
    monkeypatch.setattr(connection.server, "list_tools", list_remote)
    server = DurableMcpServer(
        connection,
        recorder=None,
        schema_cache=McpSchemaCache(),
        schema_cache_ttl_seconds=0,
        tracker=DispatchTurnTracker(),
        traced_operations=McpTracedOperations(Tracer()),
    )

    first = await server.list_tools()
    second = await server.list_tools()

    assert calls == 1
    assert [tool.name for tool in first] == ["echo"]
    assert [tool.name for tool in second] == ["echo"]


async def test_durable_server_captures_mcp_execution_without_sdk_lifecycle_hook() -> None:
    async def call_remote(
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> object:
        return SimpleNamespace(isError=False, content=[SimpleNamespace(type="text", text="Python")])

    server = DurableMcpServer(
        pooled_transport(call_remote),
        recorder=None,
        schema_cache=McpSchemaCache(),
        schema_cache_ttl_seconds=60,
        tracker=DispatchTurnTracker(),
        traced_operations=McpTracedOperations(Tracer()),
    )
    definition = ToolDefinition(
        tool_key="mcp.fixture.search",
        tool_name="fixture_search",
        description="Search fixture data.",
        parameters={"type": "object"},
        source_type=ToolSourceType.MCP,
    )
    collector = ExternalExecutionRecorder()
    server.bind_execution_collector(collector, [definition])

    result = await server.call_tool(
        "search",
        {"query": "Python"},
        {"agentbreaker/tool_call_id": "call-7", "agentbreaker/turn_number": 2},
    )

    assert isinstance(result, SimpleNamespace)
    assert result.isError is False
    assert collector.executions == [("call-7", definition, '{"query":"Python"}', "COMPLETED", result, "")]


class FakeProgressClient:
    def __init__(self) -> None:
        self.requests: list[AppendConversationRoundProgressRequest] = []

    async def append_round_progress(
        self, request: AppendConversationRoundProgressRequest
    ) -> AppendConversationRoundProgressResponse:
        self.requests.append(request)
        return AppendConversationRoundProgressResponse(
            base=ResponseBase(code=0, success=True),
            data=ConversationRoundMutationResult(committed_revision=len(self.requests)),
        )


async def test_dispatch_recorder_redacts_arguments_and_advances_revision() -> None:
    client = FakeProgressClient()
    state = SimpleNamespace(checkpoint_revision=0, revision_lock=asyncio.Lock())
    recorder = ConversationDispatchRecorder(client, state, 7, "conv_dispatch", 3)  # type: ignore[arg-type]

    await recorder.before_dispatch(
        "attempt-1",
        "call-1",
        2,
        "fixture",
        "send",
        {"api_key": "secret", "nested": {"token": "secret", "ok": True}},
    )
    await recorder.after_dispatch("attempt-1", "COMPLETED")

    assert [request.expected_revision for request in client.requests] == [0, 1]
    assert state.checkpoint_revision == 2
    first = client.requests[0].dispatch_evidence[0]
    assert first.state == ToolDispatchState.DISPATCHING
    assert first.tool_call_id == "call-1"
    assert first.turn_number == 2
    assert first.tool_key == "mcp.fixture.send"
    assert first.arguments_json == '{"api_key":"[REDACTED]","nested":{"token":"[REDACTED]","ok":true}}'
    assert client.requests[1].dispatch_evidence[0].state == ToolDispatchState.COMPLETED
