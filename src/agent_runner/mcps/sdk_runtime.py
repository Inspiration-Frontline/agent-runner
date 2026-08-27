"""Request adapters over pooled Streamable HTTP MCP connections.

The application pool owns SDK connection lifetimes. Each Agent request borrows exclusive
connections, discovers their tools, and supplies request-scoped wrappers to the OpenAI Agents SDK.
The wrappers hold dispatch evidence and model-turn metadata, never the shared transport lifecycle.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

from agents import Agent
from agents.agent import AgentBase
from agents.lifecycle import AgentHooksBase
from agents.mcp import MCPServer, MCPServerStreamableHttp, MCPUtil
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
    TaskAffineMcpConnectionManager,
)
from agent_runner.mcps.dispatch_tracing import McpTracedOperations
from agent_runner.mcps.failures import McpFailureCode, classify_mcp_failure, mcp_failure
from agent_runner.observability.tracing import Tracer
from agent_runner.tools.registry import ToolDefinition, ToolSourceType

logger = logging.getLogger(__name__)

_CALL_ID_META_KEY = "agentbreaker/tool_call_id"
_TURN_NUMBER_META_KEY = "agentbreaker/turn_number"


async def _await_cancellation_safe_cleanup(cleanup: Awaitable[object]) -> None:
    """Finish owned transport cleanup before propagating one or more request cancellations."""
    cleanup_task = asyncio.ensure_future(cleanup)
    cancelled = False
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            cancelled = True
    cleanup_error = cleanup_task.exception()
    if cleanup_error is not None:
        raise cleanup_error
    if cancelled:
        raise asyncio.CancelledError


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
        # Key: credential-isolated McpConnectionKey.cache_key. Value: Tool schema snapshot and expiry.
        self._entries: dict[str, _SchemaCacheEntry] = {}
        # Key: the same cache key. Value: per-server lock that coalesces concurrent cache refreshes.
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
        # Key: model Tool call ID. Value: durable dispatch state and recovery explanation.
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
        # Key: SDK-visible Tool name. Value: AgentBreaker definition carrying source provenance.
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
        self._request_schema_lock = asyncio.Lock()
        self._request_tools: tuple[McpTool, ...] | None = None
        self._execution_collector: ExternalExecutionCollector | None = None
        # Key: MCP protocol Tool name. Value: AgentBreaker definition with stable provenance.
        self._definitions_by_remote_name: dict[str, ToolDefinition] = {}

    def bind_execution_collector(
        self,
        collector: ExternalExecutionCollector,
        definitions: Sequence[ToolDefinition],
    ) -> None:
        """Attach request-scoped execution capture after MCP schemas receive public SDK names.

        The Agents SDK invokes MCP FunctionTools through ``call_tool`` without reliably invoking
        Agent hooks. Capturing at this protocol boundary preserves the matching public Tool
        definition for both streamed completion events and durable history.
        """
        tool_key_prefix = f"mcp.{self.name}."
        self._execution_collector = collector
        self._definitions_by_remote_name = {
            definition.tool_key.removeprefix(tool_key_prefix): definition
            for definition in definitions
            if definition.source_type == ToolSourceType.MCP
            and definition.tool_key.startswith(tool_key_prefix)
        }

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
        """Return one immutable schema snapshot for the complete Agent request."""
        async with self._request_schema_lock:
            if self._request_tools is None:
                self._request_tools = tuple(await self._discover_tools(run_context, agent))
            return list(self._request_tools)

    async def _discover_tools(
        self,
        run_context: RunContextWrapper[Any] | None,
        agent: AgentBase | None,
    ) -> list[McpTool]:
        """Discover schemas from the cross-request cache or a remote `tools/list` call."""
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
            if classify_mcp_failure(error).transport_failed:
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
        outbound_meta = dict(meta or {})
        tool_call_id = str(outbound_meta.pop(_CALL_ID_META_KEY, ""))
        turn_number = int(outbound_meta.pop(_TURN_NUMBER_META_KEY, 1))
        if not tool_call_id:
            raise RuntimeError("SDK MCP invocation is missing its model Tool call ID.")
        attempt_id = str(uuid4())
        if self._recorder is not None:
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
        except asyncio.CancelledError:
            try:
                await self._record_cancelled_result(tool_call_id, tool_name, attempt_id)
            finally:
                self._record_execution(
                    tool_call_id,
                    tool_name,
                    arguments or {},
                    "CANCELLED",
                    None,
                    "Generation cancelled.",
                )
            raise
        except BaseException as error:
            failure = classify_mcp_failure(error)
            state = "FAILED" if failure.definitely_not_delivered else "UNKNOWN"
            self._transport_failed = failure.transport_failed
            await self._record_result(tool_call_id, tool_name, attempt_id, state, failure.public_message, failure.code)
            self._record_execution(tool_call_id, tool_name, arguments or {}, state, None, failure.public_message)
            raise
        state = "FAILED" if bool(getattr(result, "isError", False)) else "COMPLETED"
        reason = "MCP Tool returned an error result." if state == "FAILED" else ""
        await self._record_result(tool_call_id, tool_name, attempt_id, state, reason)
        self._record_execution(tool_call_id, tool_name, arguments or {}, state, result, reason)
        return result

    async def _record_cancelled_result(self, tool_call_id: str, tool_name: str, attempt_id: str) -> None:
        """Finish cancellation evidence even when the SDK cancels the current Tool task."""
        record_task = asyncio.create_task(
            self._record_result(tool_call_id, tool_name, attempt_id, "CANCELLED", "Generation cancelled.")
        )
        while not record_task.done():
            try:
                await asyncio.shield(record_task)
            except asyncio.CancelledError:
                continue
        if record_task.cancelled():
            return
        error = record_task.exception()
        if error is not None:
            raise error

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
        if recorder is not None:
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

    def _record_execution(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        state: str,
        result: object,
        reason: str,
    ) -> None:
        """Record terminal MCP evidence without depending on local SDK lifecycle hooks."""
        collector = self._execution_collector
        definition = self._definitions_by_remote_name.get(tool_name)
        if collector is None or definition is None:
            return
        collector.record_external_execution(
            tool_call_id,
            definition,
            json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            state,
            result,
            reason,
        )

    async def list_prompts(self) -> Any:
        """Delegate prompt listing to the borrowed transport when an SDK caller requests it."""
        return await self._connection.server.list_prompts()

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Delegate prompt lookup to the borrowed transport when an SDK caller requests it."""
        return await self._connection.server.get_prompt(name, arguments)

@dataclass(frozen=True)
class McpConnectionDiagnostic:
    """Connection/preflight result exposed to request orchestration without leaking credentials."""

    server_id: str
    required: bool
    connected: bool
    failure_code: McpFailureCode | None = None
    message: str = ""


@dataclass(frozen=True)
class McpServerBorrowResult:
    """One binding's connection-preflight diagnostic and optional request-scoped server wrapper."""

    diagnostic: McpConnectionDiagnostic
    server: DurableMcpServer | None


@dataclass(frozen=True)
class DiscoveredMcpServer:
    """One prepared server paired with the Tool schemas discovered from it."""

    server: DurableMcpServer
    tools: tuple[McpTool, ...]


@dataclass(frozen=True)
class McpServerDiscoveryError:
    """Schema-discovery failure associated with one catalog server identity."""

    server_id: str
    failure_code: McpFailureCode
    message: str


@dataclass(frozen=True)
class McpServerDiscoveryResult:
    """Complete schema-discovery outcome without positional multi-value return semantics."""

    servers: tuple[DiscoveredMcpServer, ...]
    errors: tuple[McpServerDiscoveryError, ...]


@dataclass(frozen=True)
class ActiveMcpSession:
    """The request-scoped SDK servers, tool definitions, diagnostics, and event hooks."""

    servers: tuple[MCPServer, ...]
    diagnostics: tuple[McpConnectionDiagnostic, ...]
    definitions: tuple[ToolDefinition, ...]
    dispatch_hooks: McpDispatchHooks


class RequiredMcpServerUnavailableError(RuntimeError):
    """Raised before the LLM call when an Agent's required MCP server cannot be prepared."""

    error_code = "MCP_REQUIRED_SERVER_UNAVAILABLE"

    def __init__(self, diagnostics: Sequence[McpConnectionDiagnostic]) -> None:
        """Build a credential-safe aggregate from failed required server diagnostics."""
        details = "; ".join(f"{item.server_id}: {item.message}" for item in diagnostics)
        super().__init__(f"Required MCP server unavailable: {details}")
        self.diagnostics = tuple(diagnostics)


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
                borrow_result = await self._borrow_server(binding, recorder, tracker)
                diagnostics.append(borrow_result.diagnostic)
                if borrow_result.server is not None:
                    leases.append(borrow_result.server)
            failed_required = [item for item in diagnostics if item.required and not item.connected]
            if failed_required:
                self._trace_diagnostics(diagnostics)
                raise RequiredMcpServerUnavailableError(failed_required)
            discovery_result = await self._discover_servers(leases)
            diagnostics = self._merge_discovery_errors(diagnostics, discovery_result.errors)
            failed_required = [item for item in diagnostics if item.required and not item.connected]
            self._trace_diagnostics(diagnostics)
            if failed_required:
                raise RequiredMcpServerUnavailableError(failed_required)
            active_servers = tuple(item.server for item in discovery_result.servers)
            definitions = tuple(self._build_definitions(discovery_result.servers))
            yield ActiveMcpSession(active_servers, tuple(diagnostics), definitions, McpDispatchHooks(tracker))
        finally:
            await _await_cancellation_safe_cleanup(
                asyncio.gather(
                    *(self._connection_pool.release(server._connection, server.transport_failed) for server in leases),
                    return_exceptions=True,
                )
            )

    async def _borrow_server(
        self,
        binding: MCPServerBinding,
        recorder: DispatchEvidenceRecorder | None,
        tracker: DispatchTurnTracker,
    ) -> McpServerBorrowResult:
        """Resolve one binding into a named preflight result with an optional borrowed server."""
        try:
            resolved = self._catalog.resolve(binding.server_id)
            if not resolved.profile.enabled:
                failure = mcp_failure(McpFailureCode.SERVER_DISABLED)
                diagnostic = McpConnectionDiagnostic(
                    binding.server_id,
                    binding.required,
                    False,
                    failure.code,
                    failure.public_message,
                )
                return McpServerBorrowResult(diagnostic, None)
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
            return McpServerBorrowResult(
                McpConnectionDiagnostic(binding.server_id, binding.required, True),
                server,
            )
        except BaseException as error:
            failure = classify_mcp_failure(error)
            logger.warning(
                "MCP connection preflight failed: server_id=%s required=%s failure_code=%s message=%s",
                binding.server_id,
                binding.required,
                failure.code,
                failure.public_message,
                extra={"server_id": binding.server_id, "mcp_failure_code": failure.code},
            )
            diagnostic = McpConnectionDiagnostic(
                binding.server_id,
                binding.required,
                False,
                failure.code,
                failure.public_message,
            )
            return McpServerBorrowResult(diagnostic, None)

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
        manager = TaskAffineMcpConnectionManager(server, item.profile.connection_timeout_seconds)
        # MCPServerManager.connect_all() already cleans every partially opened server before
        # propagating an enter failure. Calling cleanup_all() again here can move AnyIO transport
        # teardown onto the request task after the manager's task-affine worker has exited.
        await manager.__aenter__()
        if not manager.active_servers:
            await manager.cleanup_all()
            raise RuntimeError(f"MCP server {item.server_id} did not produce an active connection.")
        return PooledMcpConnection(key, server, manager, monotonic())

    @staticmethod
    def _merge_discovery_errors(
        diagnostics: list[McpConnectionDiagnostic],
        errors: Sequence[McpServerDiscoveryError],
    ) -> list[McpConnectionDiagnostic]:
        """Replace successful connection diagnostics with discovery failures for the same server."""
        errors_by_server_id = {error.server_id: error for error in errors}
        return [
            McpConnectionDiagnostic(
                item.server_id,
                item.required,
                False,
                errors_by_server_id[item.server_id].failure_code,
                errors_by_server_id[item.server_id].message,
            )
            if item.server_id in errors_by_server_id
            else item
            for item in diagnostics
        ]

    @staticmethod
    async def _discover_servers(
        servers: Sequence[DurableMcpServer],
    ) -> McpServerDiscoveryResult:
        """Run concurrent `tools/list` calls and return one named aggregate discovery result."""
        results = await asyncio.gather(*(server.list_tools() for server in servers), return_exceptions=True)
        discovered: list[DiscoveredMcpServer] = []
        errors: list[McpServerDiscoveryError] = []
        for server, result in zip(servers, results, strict=True):
            if isinstance(result, BaseException):
                failure = classify_mcp_failure(result)
                errors.append(McpServerDiscoveryError(server.name, failure.code, failure.public_message))
                logger.warning(
                    "MCP schema discovery failed: server_id=%s failure_code=%s message=%s",
                    server.name,
                    failure.code,
                    failure.public_message,
                    extra={"server_id": server.name, "mcp_failure_code": failure.code},
                )
            else:
                discovered.append(DiscoveredMcpServer(server, tuple(result)))
        return McpServerDiscoveryResult(tuple(discovered), tuple(errors))

    @staticmethod
    def _build_definitions(
        discovered: Sequence[DiscoveredMcpServer],
    ) -> list[ToolDefinition]:
        """Convert discovered MCP schemas into SDK function tools plus AgentBreaker provenance."""
        batches: list[tuple[int, MCPServer, list[McpTool]]] = [
            (index, item.server, list(item.tools)) for index, item in enumerate(discovered)
        ]
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
                definitions.append(
                    ToolDefinition.from_function_tool(
                        f"mcp.{server.name}.{tool.name}",
                        function_tool,
                        ToolSourceType.MCP,
                    )
                )
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
                    "mcp.server.failure_code": diagnostic.failure_code or "",
                },
            )
