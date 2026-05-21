import time
from dataclasses import dataclass
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, generate_latest


REQUEST_COUNT = Counter(
    "agent_runner_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "agent_runner_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)

ACTIVE_REQUESTS = Gauge(
    "agent_runner_active_requests",
    "Number of active requests",
    ["endpoint"],
)

TOOL_CALLS = Counter(
    "agent_runner_tool_calls_total",
    "Total number of tool calls",
    ["tool_name", "status"],
)

MODEL_CALLS = Counter(
    "agent_runner_model_calls_total",
    "Total number of model calls",
    ["model", "status"],
)

TOKENS_USED = Counter(
    "agent_runner_tokens_total",
    "Total tokens used",
    ["model", "type"],
)


@dataclass
class MetricsCollector:
    def record_request(self, method: str, endpoint: str, status: int, latency: float):
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

    def record_tool_call(self, tool_name: str, status: str):
        TOOL_CALLS.labels(tool_name=tool_name, status=status).inc()

    def record_model_call(self, model: str, status: str):
        MODEL_CALLS.labels(model=model, status=status).inc()

    def record_tokens(self, model: str, token_type: str, count: int):
        TOKENS_USED.labels(model=model, type=token_type).inc(count)

    def get_metrics(self) -> bytes:
        return generate_latest()


_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


async def metrics_middleware(request, call_next):
    start_time = time.time()

    ACTIVE_REQUESTS.labels(endpoint=request.url.path).inc()

    try:
        response = await call_next(request)
        latency = time.time() - start_time

        get_metrics_collector().record_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            latency=latency,
        )

        return response

    finally:
        ACTIVE_REQUESTS.labels(endpoint=request.url.path).dec()
