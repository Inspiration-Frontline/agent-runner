from dataclasses import dataclass

from agent_runner.config import ConfigurationManager, Settings
from agent_runner.observability.metrics import MetricsCollector
from agent_runner.observability.tracing import TracingManager
from agent_runner.runtime.cancellation import ConversationCancellationRegistry


@dataclass(frozen=True)
class ApplicationServices:
    """Stateful service owners scoped to one FastAPI application instance."""

    configuration: ConfigurationManager
    tracing: TracingManager
    metrics: MetricsCollector
    cancellations: ConversationCancellationRegistry

    def get_settings(self) -> Settings:
        """Return the latest Nacos-over-file settings snapshot for a new request."""
        return self.configuration.get_settings()
