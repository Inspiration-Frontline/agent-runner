import re

from opentelemetry.exporter.otlp.proto.grpc import trace_exporter

from agent_runner.conversation.client import ConversationManagerClient
from agent_runner.observability import tracing

TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def test_active_span_uses_lowercase_w3c_trace_id() -> None:
    with tracing.Tracer().span("test.root") as span:
        assert TRACE_ID_PATTERN.fullmatch(span.trace_id)
        assert tracing.current_trace_id() == span.trace_id


def test_extracts_w3c_parent_context() -> None:
    expected_trace_id = "0123456789abcdef0123456789abcdef"
    expected_parent_id = "0123456789abcdef"
    parent_context = tracing.extract_trace_context(
        {"traceparent": f"00-{expected_trace_id}-{expected_parent_id}-01"}
    )

    with tracing.Tracer().span("test.child", parent_context=parent_context) as span:
        assert span.trace_id == expected_trace_id
        assert span.parent_id == expected_parent_id


def test_conversation_rpc_metadata_contains_active_w3c_context() -> None:
    with tracing.Tracer().span("test.rpc") as span:
        metadata = dict(ConversationManagerClient._get_trace_metadata())

    assert metadata["traceparent"].split("-")[1] == span.trace_id


def test_exporter_setup_failure_does_not_break_span_creation(monkeypatch) -> None:
    def fail_exporter_setup(*args, **kwargs):
        raise OSError("collector unavailable")

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", fail_exporter_setup)
    manager = tracing.TracingManager(endpoint="http://127.0.0.1:1", enabled=True)

    with manager.tracer.span("test.exporter-unavailable") as span:
        assert TRACE_ID_PATTERN.fullmatch(span.trace_id)


def test_trace_json_redacts_nested_secrets_and_preserves_token_usage() -> None:
    serialized = tracing.trace_json(
        {
            "authorization": "Bearer secret",
            "nested": '{"api_key":"hidden","value":42}',
            "prompt_tokens": 12,
        },
        max_chars=1024,
    )

    assert "Bearer secret" not in serialized
    assert "hidden" not in serialized
    assert serialized.count("[REDACTED]") == 2
    assert '"prompt_tokens":12' in serialized


def test_trace_json_applies_a_hard_length_limit() -> None:
    serialized = tracing.trace_json({"result": "x" * 1000}, max_chars=128)

    assert len(serialized) == 128
    assert "truncated" in serialized
