from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "agent_runner"


def test_gateway_litellm_client_does_not_implement_span_recording() -> None:
    source = (SOURCE_ROOT / "gateway" / "litellm_client.py").read_text(encoding="utf-8")

    assert "SpanKind" not in source
    assert "span.set_attribute" not in source
    assert "span.add_event" not in source
    assert "trace_json" not in source


def test_runtime_orchestrator_does_not_implement_span_recording() -> None:
    source = (SOURCE_ROOT / "runtime" / "orchestrator.py").read_text(encoding="utf-8")

    assert "span.set_attribute" not in source
    assert "span.add_event" not in source
    assert "trace_json" not in source
    assert "from agent_runner.observability.tracing import current_trace_id" not in source
