from dataclasses import dataclass

from agent_runner.config import ConfigurationManager, Settings
from agent_runner.mcps.connection_pool import McpConnectionPool
from agent_runner.mcps.sdk_runtime import McpSchemaCache
from agent_runner.mcps.secrets import SecretProvider
from agent_runner.observability.metrics import MetricsCollector
from agent_runner.observability.tracing import TracingManager
from agent_runner.runtime.cancellation import ConversationCancellationRegistry


@dataclass(frozen=True)
class ApplicationServices:
    """Stateful service owners scoped to one FastAPI application instance.

    Attributes:
        configuration: File and Nacos configuration owner used for request snapshots.
        tracing: OpenTelemetry provider and span factory for this application.
        metrics: Prometheus collector registry owned by this application.
        cancellations: Registry of active Conversation cancellation tokens.
        mcp_connection_pool: Bounded pool of credential-isolated MCP connections.
        mcp_schema_cache: Application cache for discovered MCP Tool schemas.
        mcp_secret_provider: Snapshot provider resolving MCP Secret references at the boundary.
    """

    configuration: ConfigurationManager
    """Configuration owner for file-backed and Nacos settings."""
    tracing: TracingManager
    """Tracing provider and application-owned OpenTelemetry facade."""
    metrics: MetricsCollector
    """Prometheus metrics registry scoped to this application instance."""
    cancellations: ConversationCancellationRegistry
    """Active request cancellation registry."""
    mcp_connection_pool: McpConnectionPool
    """Bounded pool that isolates live MCP transports by credential fingerprint."""
    mcp_schema_cache: McpSchemaCache
    """TTL cache of MCP Tool schemas discovered from remote servers."""
    mcp_secret_provider: SecretProvider
    """Provider for immutable, redacted MCP Secret snapshots."""

    def get_settings(self) -> Settings:
        """Return the latest Nacos-over-file settings snapshot for a new request.

        Returns:
            Latest Nacos-over-file settings snapshot for a new request.
        """

        return self.configuration.get_settings()
