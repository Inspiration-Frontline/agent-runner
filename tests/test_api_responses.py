from typing import Any

from agent_runner.api.debug_routes import debug_config
from agent_runner.api.responses import DebugConfigResponse, HealthResponse
from agent_runner.api.routes import ChatTraceStats
from agent_runner.api.streaming import DoneEvent, TokenDeltaEvent, UsageEvent
from agent_runner.main import app, health_check, root


class TraceSpanRecorder:
    """Minimal span double that preserves tags and event payloads for assertions."""

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any] | None]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes))


def test_api_version_matches_project_version() -> None:
    assert app.version == "0.0.1"


async def test_health_check_returns_typed_response() -> None:
    response = await health_check()

    assert response == HealthResponse(status="healthy")


async def test_root_redirects_to_swagger_docs() -> None:
    response = await root()

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


async def test_debug_config_returns_typed_response() -> None:
    response = await debug_config()

    assert isinstance(response, DebugConfigResponse)
    assert response.lite_llm_base_url == "http://localhost:4000"
    assert response.nacos_enabled is False


def test_chat_trace_stats_aggregates_multi_turn_usage_and_lifecycle() -> None:
    span = TraceSpanRecorder()
    stats = ChatTraceStats()

    stats.record(span, UsageEvent(579, 35, 614))
    stats.record(span, TokenDeltaEvent("complete answer"))
    stats.record(span, UsageEvent(660, 34, 694))
    stats.record(span, DoneEvent())
    stats.finish(span)

    assert span.attributes["gen_ai.usage.input_tokens"] == 1239
    assert span.attributes["gen_ai.usage.output_tokens"] == 69
    assert span.attributes["gen_ai.usage.total_tokens"] == 1308
    assert span.attributes["chat.response_chars"] == 15
    assert span.attributes["chat.outcome"] == "completed"
    assert span.events[0] == (
        "chat.usage",
        {
            "gen_ai.usage.input_tokens": 579,
            "gen_ai.usage.output_tokens": 35,
            "gen_ai.usage.total_tokens": 614,
        },
    )
