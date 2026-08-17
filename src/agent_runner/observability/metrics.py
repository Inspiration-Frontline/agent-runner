import time
from collections.abc import Awaitable, Callable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response


class MetricsCollector:
    """Application-scoped Prometheus instruments and HTTP metrics middleware."""

    def __init__(self) -> None:
        """Create an isolated registry so tests and multiple app instances never share collectors."""
        self._registry = CollectorRegistry()
        self._request_count = Counter(
            "agent_runner_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
            registry=self._registry,
        )
        self._request_latency = Histogram(
            "agent_runner_request_latency_seconds",
            "Request latency in seconds",
            ["method", "endpoint"],
            registry=self._registry,
        )
        self._active_requests = Gauge(
            "agent_runner_active_requests",
            "Number of active requests",
            ["endpoint"],
            registry=self._registry,
        )
        self._tool_calls = Counter(
            "agent_runner_tool_calls_total",
            "Total number of tool calls",
            ["tool_name", "status"],
            registry=self._registry,
        )
        self._model_calls = Counter(
            "agent_runner_model_calls_total",
            "Total number of model calls",
            ["model", "status"],
            registry=self._registry,
        )
        self._tokens_used = Counter(
            "agent_runner_tokens_total",
            "Total tokens used",
            ["model", "type"],
            registry=self._registry,
        )

    def record_request(self, method: str, endpoint: str, status: int, latency: float) -> None:
        """Record one completed HTTP request."""
        self._request_count.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self._request_latency.labels(method=method, endpoint=endpoint).observe(latency)

    def record_tool_call(self, tool_name: str, status: str) -> None:
        """Record one Tool terminal status."""
        self._tool_calls.labels(tool_name=tool_name, status=status).inc()

    def record_model_call(self, model: str, status: str) -> None:
        """Record one model terminal status."""
        self._model_calls.labels(model=model, status=status).inc()

    def record_tokens(self, model: str, token_type: str, count: int) -> None:
        """Record model token usage."""
        self._tokens_used.labels(model=model, type=token_type).inc(count)

    def get_metrics(self) -> bytes:
        """Render only this application instance's Prometheus registry."""
        return generate_latest(self._registry)

    async def collect_http_metrics(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Collect request count, latency, and active-request metrics."""
        start_time = time.monotonic()
        endpoint = request.url.path
        self._active_requests.labels(endpoint=endpoint).inc()
        try:
            response = await call_next(request)
            self.record_request(
                method=request.method,
                endpoint=endpoint,
                status=response.status_code,
                latency=time.monotonic() - start_time,
            )
            return response
        finally:
            self._active_requests.labels(endpoint=endpoint).dec()
