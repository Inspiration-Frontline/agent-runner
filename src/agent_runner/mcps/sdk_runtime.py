"""Request adapters over pooled Streamable HTTP MCP connections.

The application pool owns SDK connection lifetimes. Each Agent request borrows exclusive
connections, discovers their tools, and supplies request-scoped wrappers to the OpenAI Agents SDK.
The wrappers hold dispatch evidence and model-turn metadata, never the shared transport lifecycle.
"""

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
from agent_runner.config import Settings
from agent_runner.mcps.catalog import McpServerCatalog, ResolvedMcpServer
from agent_runner.mcps.connection_pool import (
    McpConnectionKey,
    McpConnectionPool,
    McpConnectionPoolSettings,
    PooledMcpConnection,
)
from agent_runner.mcps.dispatch_tracing import McpTracedOperations
from agent_runner.observability.tracing import Tracer
from agent_runner.tools.registry import ToolDefinition, ToolSourceType

logger = logging.getLogger(__name__)

_CALL_ID_META_KEY = "agentbreaker/tool_call_id"
_TURN_NUMBER_META_KEY = "agentbreaker/turn_number"


class DispatchEvidenceRecorder(Protocol):
    """Contract used to durably record a remote MCP tool delivery attempt."""

    async def before_dispatch(
        self,
        attempt_id: str,
        tool_call_id: str,
        turn_number: int,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Persist intent before bytes for the remote call are sent."""

    async def after_dispatch(self, attempt_id: str, state: str, recovery_reason: str = "") -> None:
        """Persist the terminal delivery state after a result or transport failure."""


@dataclass(frozen=True)
class _SchemaCacheEntry:
    """One immutable `tools/list` response and its monotonic expiry time."""

    expires_at: float
    tools: tuple[McpTool, ...]


class McpSchemaCache:
    """Application-scoped schema cache with a single refresh flight for each server identity."""

    def __init__(self) -> None:
        """Create an empty cache; entries are created lazily after a successful discovery."""
        self._entries: dict[str, _SchemaCacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(
        self,
        cache_key: str,
        ttl_seconds: int,
        loader: Callable[[], Awaitable[list[McpTool]]],
    ) -> list[McpTool]:
        """Return a fresh or cached schema snapshot without exposing stale tools after expiry."""
        if ttl_seconds <= 0:
            return await loader()
        lock = self._locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._entries.get(cache_key)
            now = monotonic()
            if cached is not None and cached.expires_at > now:
                return list(cached.tools)
            tools = await loader()
            self._entries[cache_key] = _SchemaCacheEntry(now + ttl_seconds, tuple(tools))
            return tools

    def invalidate(self, cache_key: str) -> None:
        """Discard a schema snapshot after a connection or protocol failure."""
        self._entries.pop(cache_key, None)


class DispatchTurnTracker:
    """Request-scoped mapping from Agents SDK hooks to durable MCP dispatch metadata."""

    def __init__(self) -> None:
        """Start an empty tracker before the first model response is received."""
        self.turn_number = 0
        self.outcomes: dict[str, tuple[str, str]] = {}

    def model_response_completed(self) -> None:
        """Advance the model-turn counter after the SDK completes one LLM response."""
        self.turn_number += 1

    def resolve_meta(self, context: Any) -> dict[str, Any] | None:
        """Produce MCP `_meta` fields for a model-selected tool call, when SDK context is present."""
        if not isinstance(context.run_context, ToolContext):
            return None
        return {
            _CALL_ID_META_KEY: context.run_context.tool_call_id,
            _TURN_NUMBER_META_KEY: max(1, self.turn_number),
        }

    def record_outcome(self, tool_call_id: str, state: str, reason: str = "") -> None:
        """Retain the durable dispatch result so the generic tool hook can persist execution evidence."""
        self.outcomes[tool_call_id] = (state, reason)


class ExternalExecutionCollector(Protocol):
    """Contract for persisting one terminal external tool execution in the current run capture."""

    def record_external_execution(
        self,
        tool_call_id: str,
        definition: ToolDefinition,
        arguments_json: str,
        status: str,
        result: object,
        error_message: str,
    ) -> None:
        """Store one terminal result emitted by the Agents SDK after an MCP tool call."""


class McpDispatchHooks(AgentHooksBase[Any, Agent[Any]]):
    """Agents SDK hooks that connect model/tool events to request-scoped MCP evidence."""

    def __init__(self, tracker: DispatchTurnTracker) -> None:
        """Create hooks for one request tracker before the Agent SDK starts execution."""
        self._tracker = tracker
        self._collector: ExternalExecutionCollector | None = None
        self._definitions: dict[str, ToolDefinition] = {}

    def bind_collector(self, collector: ExternalExecutionCollector, definitions: Sequence[ToolDefinition]) -> None:
        """Attach the capture collector after all internal and MCP tool definitions are known."""
        self._collector = collector
        self._definitions = {definition.tool_name: definition for definition in definitions}

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        response: Any,
    ) -> None:
        """Advance the durable MCP turn number after every completed model response."""
        self._tracker.model_response_completed()

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Any,
        result: object,
    ) -> None:
        """Capture a terminal result only when the completed tool originated from an MCP server."""
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


class DurableMcpServer(MCPServer):
    """Request-bound MCP adapter over one exclusively borrowed pooled connection.

    The class deliberately has no connection lifecycle. Its only mutable state is request-local
    evidence and turn metadata, so returning the transport to the application pool is safe.
    """

    def __init__(
        self,
        connection: PooledMcpConnection,
        recorder: DispatchEvidenceRecorder | None,
        schema_cache: McpSchemaCache,
        schema_cache_ttl_seconds: int,
        tracker: DispatchTurnTracker,
        traced_operations: McpTracedOperations,
    ) -> None:
        """Wrap a borrowed transport with the current request's evidence collaborators."""
        super().__init__(require_approval="never", tool_meta_resolver=tracker.resolve_meta)
        self._connection = connection
        self._recorder = recorder
        self._schema_cache = schema_cache
        self._schema_cache_ttl_seconds = schema_cache_ttl_seconds
        self._tracker = tracker
        self._traced_operations = traced_operations
        self._transport_failed = False

    @property
    def name(self) -> str:
        """Return the stable catalog server ID used by the SDK's tool-name prefix."""
        return self._connection.server.name

    @property
    def transport_failed(self) -> bool:
        """Return whether the borrowed transport must be evicted instead of returned to the pool."""
        return self._transport_failed

    async def connect(self) -> None:
        """Satisfy the SDK server contract; the pool already connected this leased transport."""

    async def cleanup(self) -> None:
        """Satisfy the SDK server contract; pool release owns real connection cleanup."""

    async def list_tools(
        self,
        run_context: RunContextWrapper[Any] | None = None,
        agent: AgentBase | None = None,
    ) -> list[McpTool]:
        """Discover this server's tools from the shared cache or through a fresh `tools/list` call."""
        async def load() -> list[McpTool]:
            return await self._traced_operations.discover(
                self.name,
                lambda: self._connection.server.list_tools(run_context, agent),
            )

        try:
            return await self._schema_cache.get(
                self._connection.key.cache_key,
                self._schema_cache_ttl_seconds,
                load,
            )
        except BaseException as error:
            if self._is_transport_failure(error):
                self._transport_failed = True
                self._schema_cache.invalidate(self._connection.key.cache_key)
            raise

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        """Persist dispatch intent, deliver one remote tool call, then persist its terminal evidence."""
        if self._recorder is None:
            return await self._connection.server.call_tool(tool_name, arguments, meta)
        outbound_meta = dict(meta or {})
        tool_call_id = str(outbound_meta.pop(_CALL_ID_META_KEY, ""))
        turn_number = int(outbound_meta.pop(_TURN_NUMBER_META_KEY, 1))
        if not tool_call_id:
            raise RuntimeError("SDK MCP invocation is missing its model Tool call ID.")
        attempt_id = str(uuid4())
        await self._recorder.before_dispatch(
            attempt_id,
            tool_call_id,
            turn_number,
            self.name,
            tool_name,
            arguments or {},
        )
        try:
            result = await self._traced_operations.dispatch(
                self.name,
                tool_name,
                tool_call_id,
                attempt_id,
                lambda: self._connection.server.call_tool(tool_name, arguments, outbound_meta or None),
            )
        except BaseException as error:
            state = "FAILED" if self._is_definite_pre_delivery_failure(error) else "UNKNOWN"
            self._transport_failed = self._is_transport_failure(error)
            await self._record_result(tool_call_id, tool_name, attempt_id, state, str(error), type(error).__name__)
            raise
        state = "FAILED" if bool(getattr(result, "isError", False)) else "COMPLETED"
        reason = "MCP Tool returned an error result." if state == "FAILED" else ""
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
        """Persist and expose a terminal delivery outcome after the remote call has finished."""
        recorder = self._recorder
        if recorder is None:
            raise RuntimeError("Durable MCP result recording requires a dispatch recorder.")
        await self._traced_operations.record_result(
            self.name,
            tool_name,
            tool_call_id,
            attempt_id,
            state,
            lambda: recorder.after_dispatch(attempt_id, state, reason),
            error_type,
        )
        self._tracker.record_outcome(tool_call_id, state, reason)

    async def list_prompts(self) -> Any:
        """Delegate prompt listing to the borrowed transport when an SDK caller requests it."""
        return await self._connection.server.list_prompts()

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Delegate prompt lookup to the borrowed transport when an SDK caller requests it."""
        return await self._connection.server.get_prompt(name, arguments)

    @classmethod
    def _is_definite_pre_delivery_failure(cls, error: BaseException) -> bool:
        """Classify connect failures as definitely not delivered, leaving ambiguous failures UNKNOWN."""
        current: BaseException | None = error
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if isinstance(current, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
                return True
            current = current.__cause__ or current.__context__
        return False

    @classmethod
    def _is_transport_failure(cls, error: BaseException) -> bool:
        """Return whether an exception makes a pooled HTTP session unsafe to reuse."""
        current: BaseException | None = error
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if isinstance(current, (httpx.TransportError, TimeoutError, ConnectionError)):
                return True
            current = current.__cause__ or current.__context__
        return False


@dataclass(frozen=True)
class McpConnectionDiagnostic:
    """Connection/preflight result exposed to request orchestration without leaking credentials."""

    server_id: str
    required: bool
    connected: bool
    error: str = ""


@dataclass(frozen=True)
class ActiveMcpSession:
    """The request-scoped SDK servers, tool definitions, diagnostics, and event hooks."""

    servers: tuple[MCPServer, ...]
    diagnostics: tuple[McpConnectionDiagnostic, ...]
    definitions: tuple[ToolDefinition, ...]
    dispatch_hooks: McpDispatchHooks


class RequiredMcpServerUnavailableError(RuntimeError):
    """Raised before the LLM call when an Agent's required MCP server cannot be prepared."""


class SdkMcpRuntime:
    """Builds request-scoped MCP SDK views over application-owned pooled connections."""

    def __init__(
        self,
        catalog: McpServerCatalog,
        tracer: Tracer | None = None,
        connection_pool: McpConnectionPool | None = None,
        settings: Settings | None = None,
        schema_cache: McpSchemaCache | None = None,
    ) -> None:
        """Create a runtime from a validated catalog and optional shared application services."""
        self._catalog = catalog
        self._schema_cache = schema_cache or McpSchemaCache()
        self._tracer = tracer or Tracer()
        self._traced_operations = McpTracedOperations(self._tracer)
        self._connection_pool = connection_pool or McpConnectionPool()
        self._settings = settings or Settings()

    @asynccontextmanager
    async def session(
        self,
        bindings: list[MCPServerBinding],
        recorder: DispatchEvidenceRecorder | None = None,
    ) -> AsyncIterator[ActiveMcpSession]:
        """Borrow bound servers, discover schemas, and release every lease on normal exit or cancellation."""
        tracker = DispatchTurnTracker()
        leases: list[DurableMcpServer] = []
        diagnostics: list[McpConnectionDiagnostic] = []
        try:
            for binding in bindings:
                diagnostic, server = await self._borrow_server(binding, recorder, tracker)
                diagnostics.append(diagnostic)
                if server is not None:
                    leases.append(server)
            failed_required = [item for item in diagnostics if item.required and not item.connected]
            if failed_required:
                details = "; ".join(f"{item.server_id}: {item.error}" for item in failed_required)
                raise RequiredMcpServerUnavailableError(f"Required MCP server unavailable: {details}")
            discovered, discovery_errors = await self._discover_servers(leases)
            diagnostics = self._merge_discovery_errors(diagnostics, discovery_errors)
            failed_required = [item for item in diagnostics if item.required and not item.connected]
            self._trace_diagnostics(diagnostics)
            if failed_required:
                details = "; ".join(f"{item.server_id}: {item.error}" for item in failed_required)
                raise RequiredMcpServerUnavailableError(f"Required MCP server unavailable: {details}")
            active_servers = tuple(server for server, _ in discovered)
            definitions = tuple(self._build_definitions(discovered))
            yield ActiveMcpSession(active_servers, tuple(diagnostics), definitions, McpDispatchHooks(tracker))
        finally:
            await asyncio.gather(
                *(self._connection_pool.release(server._connection, server.transport_failed) for server in leases),
                return_exceptions=True,
            )

    async def _borrow_server(
        self,
        binding: MCPServerBinding,
        recorder: DispatchEvidenceRecorder | None,
        tracker: DispatchTurnTracker,
    ) -> tuple[McpConnectionDiagnostic, DurableMcpServer | None]:
        """Resolve one binding and borrow an exclusive connection or return a non-secret diagnostic."""
        try:
            resolved = self._catalog.resolve(binding.server_id)
            if not resolved.profile.enabled:
                return McpConnectionDiagnostic(binding.server_id, binding.required, False, "Server is disabled"), None
            key = McpConnectionKey.from_resolved(resolved)
            connection = await self._connection_pool.borrow(
                key,
                self._pool_settings(),
                lambda: self._create_connection(key, resolved),
            )
            server = DurableMcpServer(
                connection,
                recorder,
                self._schema_cache,
                resolved.profile.schema_cache_ttl_seconds,
                tracker,
                self._traced_operations,
            )
            return McpConnectionDiagnostic(binding.server_id, binding.required, True), server
        except BaseException as error:
            logger.warning("MCP connection preflight failed", extra={"server_id": binding.server_id, "error": str(error)})
            return McpConnectionDiagnostic(binding.server_id, binding.required, False, str(error) or type(error).__name__), None

    def _pool_settings(self) -> McpConnectionPoolSettings:
        """Translate the current Nacos-over-file settings snapshot into pool limits."""
        return McpConnectionPoolSettings(
            max_connections_per_server=self._settings.mcp_pool_max_connections_per_server,
            idle_timeout_seconds=self._settings.mcp_pool_idle_timeout_seconds,
            borrow_timeout_seconds=self._settings.mcp_pool_borrow_timeout_seconds,
        )

    async def _create_connection(self, key: McpConnectionKey, item: ResolvedMcpServer) -> PooledMcpConnection:
        """Open one SDK manager/server pair after the pool has reserved capacity for its key."""
        server = MCPServerStreamableHttp(
            params={
                "url": item.url,
                "headers": item.headers,
                "timeout": item.profile.request_timeout_seconds,
                "sse_read_timeout": item.profile.request_timeout_seconds,
            },
            name=item.server_id,
            cache_tools_list=False,
            client_session_timeout_seconds=item.profile.request_timeout_seconds,
            tool_filter={"blocked_tool_names": sorted(item.profile.disabled_tools)},
            max_retry_attempts=0,
            retry_backoff_seconds_base=item.profile.retry_backoff_seconds,
            require_approval="never",
        )
        manager = MCPServerManager(
            [server],
            connect_timeout_seconds=item.profile.connection_timeout_seconds,
            cleanup_timeout_seconds=10,
            drop_failed_servers=True,
            strict=True,
            connect_in_parallel=True,
        )
        try:
            await manager.__aenter__()
        except BaseException:
            await manager.cleanup_all()
            raise
        if not manager.active_servers:
            await manager.cleanup_all()
            raise RuntimeError(f"MCP server {item.server_id} did not produce an active connection.")
        return PooledMcpConnection(key, server, manager, monotonic())

    @staticmethod
    def _merge_discovery_errors(
        diagnostics: list[McpConnectionDiagnostic],
        errors: dict[str, str],
    ) -> list[McpConnectionDiagnostic]:
        """Replace successful connection diagnostics with discovery failures for the same server."""
        return [
            McpConnectionDiagnostic(item.server_id, item.required, False, errors[item.server_id])
            if item.server_id in errors
            else item
            for item in diagnostics
        ]

    @staticmethod
    async def _discover_servers(
        servers: Sequence[DurableMcpServer],
    ) -> tuple[list[tuple[DurableMcpServer, list[McpTool]]], dict[str, str]]:
        """Run `tools/list` concurrently for the currently borrowed server wrappers."""
        results = await asyncio.gather(*(server.list_tools() for server in servers), return_exceptions=True)
        discovered: list[tuple[DurableMcpServer, list[McpTool]]] = []
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
        discovered: list[tuple[DurableMcpServer, list[McpTool]]],
    ) -> list[ToolDefinition]:
        """Convert discovered MCP schemas into SDK function tools plus AgentBreaker provenance."""
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
        """Add redacted connection state events to the current request span."""
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
