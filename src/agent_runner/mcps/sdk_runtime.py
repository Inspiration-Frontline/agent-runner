import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

import httpx
from agents import Agent
from agents.agent import AgentBase
from agents.lifecycle import AgentHooksBase
from agents.mcp import MCPServer, MCPServerManager, MCPServerStreamableHttp, MCPUtil
from agents.run_context import RunContextWrapper
from agents.tool_context import ToolContext
from mcp.types import Tool as McpTool
from opentelemetry import trace

from agent_runner.agent_definitions.config_models import MCPServerBinding
from agent_runner.mcps.catalog import McpServerCatalog, ResolvedMcpServer
from agent_runner.observability.tracing import Tracer
from agent_runner.tools.registry import ToolDefinition, ToolSourceType

logger = logging.getLogger(__name__)

_CALL_ID_META_KEY = "agentbreaker/tool_call_id"
_TURN_NUMBER_META_KEY = "agentbreaker/turn_number"


class DispatchEvidenceRecorder(Protocol):
    async def before_dispatch(
        self,
        attempt_id: str,
        tool_call_id: str,
        turn_number: int,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Commit dispatch intent before network delivery."""

    async def after_dispatch(self, attempt_id: str, state: str, recovery_reason: str = "") -> None:
        """Commit terminal dispatch evidence after the remote outcome."""


@dataclass(frozen=True)
class _SchemaCacheEntry:
    expires_at: float
    tools: tuple[McpTool, ...]


class McpSchemaCache:
    """Runtime-scoped MCP schema cache with one refresh flight per Server."""

    def __init__(self) -> None:
        self._entries: dict[str, _SchemaCacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(
        self,
        server_id: str,
        ttl_seconds: int,
        loader: Callable[[], Awaitable[list[McpTool]]],
    ) -> list[McpTool]:
        if ttl_seconds <= 0:
            return await loader()
        lock = self._locks.setdefault(server_id, asyncio.Lock())
        async with lock:
            cached = self._entries.get(server_id)
            now = monotonic()
            if cached is not None and cached.expires_at > now:
                return list(cached.tools)
            tools = await loader()
            self._entries[server_id] = _SchemaCacheEntry(now + ttl_seconds, tuple(tools))
            return tools

    def invalidate(self, server_id: str) -> None:
        self._entries.pop(server_id, None)


class DispatchTurnTracker:
    """Request-scoped bridge from SDK model hooks to MCP invocation metadata."""

    def __init__(self) -> None:
        self.turn_number = 0
        self.outcomes: dict[str, tuple[str, str]] = {}

    def model_response_completed(self) -> None:
        self.turn_number += 1

    def resolve_meta(self, context: Any) -> dict[str, Any] | None:
        if not isinstance(context.run_context, ToolContext):
            return None
        return {
            _CALL_ID_META_KEY: context.run_context.tool_call_id,
            _TURN_NUMBER_META_KEY: max(1, self.turn_number),
        }

    def record_outcome(self, tool_call_id: str, state: str, reason: str = "") -> None:
        self.outcomes[tool_call_id] = (state, reason)


class ExternalExecutionCollector(Protocol):
    def record_external_execution(
        self,
        tool_call_id: str,
        definition: ToolDefinition,
        arguments_json: str,
        status: str,
        result: object,
        error_message: str,
    ) -> None:
        """Store one external SDK Tool terminal result."""


class McpDispatchHooks(AgentHooksBase[Any, Agent[Any]]):
    def __init__(self, tracker: DispatchTurnTracker) -> None:
        self._tracker = tracker
        self._collector: ExternalExecutionCollector | None = None
        self._definitions: dict[str, ToolDefinition] = {}

    def bind_collector(self, collector: ExternalExecutionCollector, definitions: Sequence[ToolDefinition]) -> None:
        self._collector = collector
        self._definitions = {definition.tool_name: definition for definition in definitions}

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        response: Any,
    ) -> None:
        self._tracker.model_response_completed()

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Any,
        result: object,
    ) -> None:
        if not isinstance(context, ToolContext) or self._collector is None:
            return
        definition = self._definitions.get(context.tool_name)
        if definition is None or definition.source_type != ToolSourceType.MCP:
            return
        state, reason = self._tracker.outcomes.get(context.tool_call_id, ("FAILED", "Missing MCP dispatch outcome."))
        self._collector.record_external_execution(
            context.tool_call_id,
            definition,
            context.tool_arguments,
            state,
            getattr(result, "output", result),
            reason,
        )


class DurableMcpServer(MCPServerStreamableHttp):
    def __init__(
        self,
        *args: Any,
        server_id: str,
        recorder: DispatchEvidenceRecorder | None,
        schema_cache: McpSchemaCache,
        schema_cache_ttl_seconds: int,
        tracker: DispatchTurnTracker,
        tracer: Tracer | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.server_id = server_id
        self.recorder = recorder
        self._schema_cache = schema_cache
        self._schema_cache_ttl_seconds = schema_cache_ttl_seconds
        self._tracker = tracker
        self._tracer = tracer or Tracer()

    async def list_tools(
        self,
        run_context: RunContextWrapper[Any] | None = None,
        agent: AgentBase | None = None,
    ) -> list[McpTool]:
        async def load() -> list[McpTool]:
            with self._tracer.span("mcp.discovery") as span:
                span.set_attribute("mcp.server.id", self.server_id)
                tools = await super(DurableMcpServer, self).list_tools(run_context, agent)
                span.set_attribute("mcp.tool.count", len(tools))
                return tools

        return await self._schema_cache.get(
            self.server_id,
            self._schema_cache_ttl_seconds,
            load,
        )

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        if self.recorder is None:
            return await super().call_tool(tool_name, arguments, meta)
        outbound_meta = dict(meta or {})
        tool_call_id = str(outbound_meta.pop(_CALL_ID_META_KEY, ""))
        turn_number = int(outbound_meta.pop(_TURN_NUMBER_META_KEY, 1))
        if not tool_call_id:
            raise RuntimeError("SDK MCP invocation is missing its model Tool call ID.")
        attempt_id = str(uuid4())
        await self.recorder.before_dispatch(
            attempt_id,
            tool_call_id,
            turn_number,
            self.server_id,
            tool_name,
            arguments or {},
        )
        with self._tracer.span("mcp.dispatch") as span:
            span.set_attribute("gen_ai.tool.call.id", tool_call_id)
            span.set_attribute("mcp.server.id", self.server_id)
            span.set_attribute("mcp.tool.name", tool_name)
            span.set_attribute("mcp.dispatch.attempt_id", attempt_id)
            try:
                result = await super().call_tool(tool_name, arguments, outbound_meta or None)
            except BaseException as error:
                state = "FAILED" if self._is_definite_pre_delivery_failure(error) else "UNKNOWN"
                span.set_attribute("mcp.dispatch.state", state)
                await self._record_result(tool_call_id, tool_name, attempt_id, state, str(error), type(error).__name__)
                raise
            state = "FAILED" if bool(getattr(result, "isError", False)) else "COMPLETED"
            reason = "MCP Tool returned an error result." if state == "FAILED" else ""
            span.set_attribute("mcp.dispatch.state", state)
            await self._record_result(tool_call_id, tool_name, attempt_id, state, reason)
            return result

    async def _record_result(
        self,
        tool_call_id: str,
        tool_name: str,
        attempt_id: str,
        state: str,
        reason: str,
        error_type: str = "",
    ) -> None:
        recorder = self.recorder
        if recorder is None:
            raise RuntimeError("Durable MCP result recording requires a dispatch recorder.")
        with self._tracer.span("mcp.result") as span:
            span.set_attribute("gen_ai.tool.call.id", tool_call_id)
            span.set_attribute("mcp.server.id", self.server_id)
            span.set_attribute("mcp.tool.name", tool_name)
            span.set_attribute("mcp.dispatch.attempt_id", attempt_id)
            span.set_attribute("mcp.result.state", state)
            if error_type:
                span.set_attribute("error.type", error_type)
            await recorder.after_dispatch(attempt_id, state, reason)
            self._tracker.record_outcome(tool_call_id, state, reason)

    @classmethod
    def _is_definite_pre_delivery_failure(cls, error: BaseException) -> bool:
        current: BaseException | None = error
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if isinstance(current, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
                return True
            current = current.__cause__ or current.__context__
        return False


@dataclass(frozen=True)
class McpConnectionDiagnostic:
    server_id: str
    required: bool
    connected: bool
    error: str = ""


@dataclass(frozen=True)
class ActiveMcpSession:
    servers: tuple[MCPServer, ...]
    diagnostics: tuple[McpConnectionDiagnostic, ...]
    definitions: tuple[ToolDefinition, ...]
    dispatch_hooks: McpDispatchHooks


class RequiredMcpServerUnavailableError(RuntimeError):
    pass


class SdkMcpRuntime:
    def __init__(self, catalog: McpServerCatalog, tracer: Tracer | None = None) -> None:
        self._catalog = catalog
        self._schema_cache = McpSchemaCache()
        self._tracer = tracer or Tracer()

    @asynccontextmanager
    async def session(
        self,
        bindings: list[MCPServerBinding],
        recorder: DispatchEvidenceRecorder | None = None,
    ) -> AsyncIterator[ActiveMcpSession]:
        resolved = [self._catalog.resolve(binding.server_id) for binding in bindings]
        tracker = DispatchTurnTracker()
        servers = [self._build_server(item, recorder, tracker) for item in resolved if item.profile.enabled]
        manager = MCPServerManager(
            servers,
            connect_timeout_seconds=max((item.profile.connection_timeout_seconds for item in resolved), default=10),
            cleanup_timeout_seconds=10,
            drop_failed_servers=True,
            strict=False,
            connect_in_parallel=True,
        )
        async with manager:
            with self._tracer.span(
                "mcp.preflight",
                {
                    "mcp.server.binding_count": len(bindings),
                    "mcp.server.required_count": sum(binding.required for binding in bindings),
                },
            ) as span:
                diagnostics = self._diagnostics(bindings, servers, manager)
                discovered, discovery_errors = await self._discover_servers(manager.active_servers)
                if discovery_errors:
                    diagnostics = self._merge_discovery_errors(diagnostics, discovery_errors)
                failed_required = [item for item in diagnostics if item.required and not item.connected]
                span.set_attribute("mcp.server.connected_count", sum(item.connected for item in diagnostics))
                span.set_attribute("mcp.preflight.status", "failed" if failed_required else "ready")
                self._trace_diagnostics(diagnostics)
                if failed_required:
                    details = "; ".join(f"{item.server_id}: {item.error}" for item in failed_required)
                    raise RequiredMcpServerUnavailableError(f"Required MCP server unavailable: {details}")
                active_servers = tuple(server for server, _ in discovered)
                definitions = tuple(self._build_definitions(discovered))
            yield ActiveMcpSession(active_servers, tuple(diagnostics), definitions, McpDispatchHooks(tracker))

    @staticmethod
    def _merge_discovery_errors(
        diagnostics: list[McpConnectionDiagnostic],
        errors: dict[str, str],
    ) -> list[McpConnectionDiagnostic]:
        return [
            McpConnectionDiagnostic(
                server_id=item.server_id,
                required=item.required,
                connected=False,
                error=errors[item.server_id],
            ) if item.server_id in errors else item
            for item in diagnostics
        ]

    def _build_server(
        self,
        item: ResolvedMcpServer,
        recorder: DispatchEvidenceRecorder | None,
        tracker: DispatchTurnTracker,
    ) -> MCPServerStreamableHttp:
        return DurableMcpServer(
            params={
                "url": item.url,
                "headers": item.headers,
                "timeout": item.profile.request_timeout_seconds,
                "sse_read_timeout": item.profile.request_timeout_seconds,
            },
            name=item.server_id,
            server_id=item.server_id,
            recorder=recorder,
            schema_cache=self._schema_cache,
            schema_cache_ttl_seconds=item.profile.schema_cache_ttl_seconds,
            tracker=tracker,
            tracer=self._tracer,
            cache_tools_list=False,
            client_session_timeout_seconds=item.profile.request_timeout_seconds,
            tool_filter={"blocked_tool_names": sorted(item.profile.disabled_tools)},
            max_retry_attempts=0,
            retry_backoff_seconds_base=item.profile.retry_backoff_seconds,
            require_approval="never",
            tool_meta_resolver=tracker.resolve_meta,
        )

    @staticmethod
    async def _discover_servers(
        servers: Sequence[MCPServer],
    ) -> tuple[list[tuple[MCPServer, list[McpTool]]], dict[str, str]]:
        results = await asyncio.gather(*(server.list_tools() for server in servers), return_exceptions=True)
        discovered: list[tuple[MCPServer, list[McpTool]]] = []
        errors: dict[str, str] = {}
        for server, result in zip(servers, results, strict=True):
            if isinstance(result, BaseException):
                errors[server.name] = str(result) or type(result).__name__
                logger.warning("MCP schema discovery failed", extra={"server_id": server.name, "error": errors[server.name]})
            else:
                discovered.append((server, result))
        return discovered, errors

    @staticmethod
    def _build_definitions(
        discovered: list[tuple[MCPServer, list[McpTool]]],
    ) -> list[ToolDefinition]:
        batches = [(index, server, tools) for index, (server, tools) in enumerate(discovered)]
        overrides = MCPUtil._build_prefixed_tool_name_overrides(batches, reserved_names=set())
        definitions: list[ToolDefinition] = []
        for server_index, server, tools in batches:
            for tool_index, tool in enumerate(tools):
                function_tool = MCPUtil.to_function_tool(
                    tool,
                    server,
                    False,
                    tool_name_override=overrides[(server_index, tool_index)],
                )
                definitions.append(ToolDefinition.from_function_tool(
                    f"mcp.{server.name}.{tool.name}",
                    function_tool,
                    ToolSourceType.MCP,
                ))
        return definitions

    @staticmethod
    def _trace_diagnostics(diagnostics: Sequence[McpConnectionDiagnostic]) -> None:
        current_span = trace.get_current_span()
        for diagnostic in diagnostics:
            current_span.add_event(
                "mcp.server.preflight",
                {
                    "mcp.server.id": diagnostic.server_id,
                    "mcp.server.required": diagnostic.required,
                    "mcp.server.connected": diagnostic.connected,
                    "mcp.server.error": diagnostic.error[:500],
                },
            )

    @staticmethod
    def _diagnostics(
        bindings: list[MCPServerBinding],
        servers: Sequence[MCPServer],
        manager: MCPServerManager,
    ) -> list[McpConnectionDiagnostic]:
        by_name = {server.name: server for server in servers}
        active = set(manager.active_servers)
        diagnostics: list[McpConnectionDiagnostic] = []
        for binding in bindings:
            server = by_name.get(binding.server_id)
            error = manager.errors.get(server) if server is not None else None
            diagnostics.append(McpConnectionDiagnostic(
                server_id=binding.server_id,
                required=binding.required,
                connected=server in active if server is not None else False,
                error=str(error or "Server is disabled"),
            ))
        return diagnostics
