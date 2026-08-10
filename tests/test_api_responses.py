from typing import Any

from fastapi.testclient import TestClient

from agent_runner.api.streaming import DoneEvent, TokenDeltaEvent, UsageEvent
from agent_runner.main import create_app
from agent_runner.observability.conversation_tracing import ConversationTraceStats


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
    app = create_app()
    assert app.version == "0.0.1"


def test_health_root_and_debug_routes() -> None:
    app = create_app()
    with TestClient(app) as client:
        health_response = client.get("/health")
        root_response = client.get("/", follow_redirects=False)
        debug_response = client.get("/v1/agent/debug/config")

    assert health_response.json() == {"status": "healthy"}
    assert root_response.status_code == 307
    assert root_response.headers["location"] == "/docs"
    assert debug_response.status_code == 200
    assert debug_response.json()["nacos_enabled"] is False


def test_conversation_trace_stats_aggregates_multi_turn_usage_and_lifecycle() -> None:
    span = TraceSpanRecorder()
    stats = ConversationTraceStats()

    stats.record_event(span, UsageEvent(579, 35, 614))
    stats.record_event(span, TokenDeltaEvent("complete answer"))
    stats.record_event(span, UsageEvent(660, 34, 694))
    stats.record_event(span, DoneEvent())
    stats.finish(span)

    assert span.attributes["gen_ai.usage.input_tokens"] == 1239
    assert span.attributes["gen_ai.usage.output_tokens"] == 69
    assert span.attributes["gen_ai.usage.total_tokens"] == 1308
    assert span.attributes["conversation.response_chars"] == 15
    assert span.attributes["conversation.outcome"] == "completed"
    assert span.events[0] == (
        "conversation.usage",
        {
            "gen_ai.usage.input_tokens": 579,
            "gen_ai.usage.output_tokens": 35,
            "gen_ai.usage.total_tokens": 614,
        },
    )
