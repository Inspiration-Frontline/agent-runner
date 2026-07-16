import asyncio
import json
from types import SimpleNamespace

import pytest
from agents.stream_events import RunItemStreamEvent

from agent_runner.runtime.cancellation import CancellationToken
from agent_runner.runtime.openai_agents_runtime import OpenAIAgentsRuntime
from agent_runner.runtime.tool_loop import CapturedToolCall, ToolExecutionCollector
from agent_runner.tools.executor import ToolExecutor
from agent_runner.tools.internal.calculator import CalculatorTool
from agent_runner.tools.internal.current_time import CurrentTimeTool
from agent_runner.tools.internal.weather import WeatherTool
from agent_runner.tools.internal.web_search import WebSearchTool, _DuckDuckGoResultParser, _VisibleTextParser
from agent_runner.tools.registry import BaseTool, ToolRegistry


class _DelayTool(BaseTool):
    def __init__(self, tool_key: str, delay: float, fails: bool = False) -> None:
        self._tool_key = tool_key
        self.delay = delay
        self.fails = fails

    @property
    def tool_key(self) -> str:
        return self._tool_key

    @property
    def tool_name(self) -> str:
        return self._tool_key.replace(".", "_")

    @property
    def description(self) -> str:
        return "Test Tool"

    async def execute(self, value: int) -> dict:
        await asyncio.sleep(self.delay)
        if self.fails:
            raise ValueError(f"failed:{value}")
        return {"value": value}


async def test_current_time_includes_timezone_offset_and_time() -> None:
    result = await CurrentTimeTool().execute("Asia/Shanghai")

    assert result["timezone"] == "Asia/Shanghai"
    assert result["utc_offset"] == "+08:00"
    assert "+08:00" in result["local_datetime"]
    assert isinstance(result["unix_timestamp"], int)


async def test_current_time_rejects_unknown_timezone() -> None:
    with pytest.raises(ValueError, match="Unknown IANA timezone"):
        await CurrentTimeTool().execute("Invalid/AgentBreaker")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(12.5 + 7.5) / 4", 5.0),
        ("-3 * (+2 + 4)", -18),
    ],
)
async def test_calculator_supports_phase5_grammar(expression: str, expected: float) -> None:
    result = await CalculatorTool().execute(expression)
    assert result == {"expression": expression, "result": expected}


async def test_calculator_rejects_code_and_reports_division_by_zero() -> None:
    with pytest.raises(ValueError, match="unsupported syntax"):
        await CalculatorTool().execute("pow(2, 8)")
    with pytest.raises(ZeroDivisionError):
        await CalculatorTool().execute("1 / 0")


async def test_weather_uses_geocoding_and_current_weather(monkeypatch) -> None:
    responses = [
        _JsonResponse({
            "results": [{
                "name": "Shanghai",
                "admin1": "Shanghai",
                "country": "China",
                "latitude": 31.22,
                "longitude": 121.46,
            }]
        }),
        _JsonResponse({
            "timezone": "Asia/Shanghai",
            "timezone_abbreviation": "GMT+8",
            "current": {"temperature_2m": 30.2},
            "current_units": {"temperature_2m": "C"},
        }),
    ]
    client = _GetOnlyClient(responses)
    monkeypatch.setattr("agent_runner.tools.internal.weather.httpx.AsyncClient", lambda **_: client)

    result = await WeatherTool().execute(" Shanghai ")

    assert result["location"]["name"] == "Shanghai"
    assert result["timezone"] == "Asia/Shanghai"
    assert result["current"]["temperature_2m"] == 30.2
    assert len(client.calls) == 2


async def test_web_search_parses_duckduckgo_and_keeps_partial_page_failures(monkeypatch) -> None:
    search_html = """
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">A title</a>
        <div class="result__snippet">A snippet</div>
        <a class="result__a" href="https://example.com/b">B title</a>
        <div class="result__snippet">B snippet</div>
    """
    client = _GetOnlyClient([_TextResponse(search_html)])
    monkeypatch.setattr("agent_runner.tools.internal.web_search.httpx.AsyncClient", lambda **_: client)

    async def fetch_result(_client, result: dict) -> dict:
        if result["title"].strip() == "A title":
            return {**result, "content": "Readable A", "error": ""}
        return {**result, "content": "", "error": "page failed"}

    tool = WebSearchTool()
    monkeypatch.setattr(tool, "_fetch_result", fetch_result)

    result = await tool.execute(" AgentBreaker ")

    assert result["query"] == "AgentBreaker"
    assert len(result["results"]) == 2
    assert result["results"][0]["url"] == "https://example.com/a"
    assert result["results"][1]["error"] == "page failed"


def test_html_parsers_close_open_results_and_remove_hidden_content() -> None:
    search_parser = _DuckDuckGoResultParser(1)
    search_parser.feed('<a class="result__a" href="https://example.com">Title</a>')
    search_parser.close()
    assert search_parser.results == [{"url": "https://example.com", "title": "Title", "snippet": ""}]

    page_parser = _VisibleTextParser()
    page_parser.feed("<main>Visible<script>hidden()</script><style>hidden</style> text</main>")
    page_parser.close()
    assert page_parser.text() == "Visible\ntext"


async def test_batch_execution_is_parallel_and_partial_failure_does_not_cancel_sibling() -> None:
    registry = ToolRegistry()
    registry.register(_DelayTool("test.success", 0.05).to_definition())
    registry.register(_DelayTool("test.failure", 0.05, fails=True).to_definition())
    executor = ToolExecutor(registry)
    started = asyncio.get_running_loop().time()

    results = await executor.execute_batch([
        {"tool_key": "test.success", "arguments": {"value": 1}},
        {"tool_key": "test.failure", "arguments": {"value": 2}},
    ])

    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 0.09
    assert results == [
        {"tool_key": "test.success", "status": "success", "result": {"value": 1}},
        {"tool_key": "test.failure", "status": "error", "error": "failed:2"},
    ]


async def test_batch_cancellation_cancels_every_in_flight_tool() -> None:
    registry = ToolRegistry()
    registry.register(_DelayTool("test.first", 10).to_definition())
    registry.register(_DelayTool("test.second", 10).to_definition())
    token = CancellationToken()
    task = asyncio.create_task(ToolExecutor(registry).execute_batch([
        {"tool_key": "test.first", "arguments": {"value": 1}},
        {"tool_key": "test.second", "arguments": {"value": 2}},
    ], token))
    await asyncio.sleep(0)

    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_collector_returns_structured_failure_for_model_and_audit() -> None:
    definition = _DelayTool("test.failure", 0, fails=True).to_definition()
    registry = ToolRegistry()
    registry.register(definition)
    collector = ToolExecutionCollector()

    model_result = await collector.execute(
        tool_call_id="call-1",
        definition=definition,
        arguments_json='{"value":2}',
        executor=ToolExecutor(registry),
        cancellation_token=None,
    )

    assert json.loads(model_result) == {"status": "error", "error": "failed:2"}
    assert collector.get("call-1").status == "FAILED"
    assert collector.get("call-1").raw_result == model_result


async def test_cancelled_observed_call_builds_partial_turn_when_sdk_removes_raw_response() -> None:
    definition = _DelayTool("test.cancel", 10).to_definition()
    registry = ToolRegistry()
    registry.register(definition)
    collector = ToolExecutionCollector()
    collector.record_call(CapturedToolCall(
        tool_call_id="call-cancel",
        tool_name=definition.tool_name,
        arguments='{"value":1}',
    ))
    token = CancellationToken()
    task = asyncio.create_task(collector.execute(
        tool_call_id="call-cancel",
        definition=definition,
        arguments_json='{"value":1}',
        executor=ToolExecutor(registry),
        cancellation_token=token,
    ))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    capture = OpenAIAgentsRuntime(model_factory=SimpleNamespace())._build_observed_partial_capture(
        initial_messages=[{"role": "user", "content": "cancel"}],
        definitions=[definition],
        collector=collector,
        run_start=1_700_000_000_000,
        trace_id="trace",
        model_completed_times=[1_700_000_000_010],
        model_completed_usages=[(10, 2, 12)],
        cancelled=True,
    )

    assert len(capture.turns) == 1
    assert capture.turns[0].response_tool_calls[0].tool_call_id == "call-cancel"
    assert capture.turns[0].tool_executions[0].status == "CANCELLED"
    assert capture.turns[0].total_tokens == 12


def test_tool_stream_events_keep_call_identity_and_status() -> None:
    runtime = OpenAIAgentsRuntime(model_factory=SimpleNamespace())
    collector = ToolExecutionCollector()
    collector._executions["call-1"] = SimpleNamespace(tool_name="test_success", status="FAILED")
    called = RunItemStreamEvent(
        name="tool_called",
        item=SimpleNamespace(
            call_id="call-1",
            raw_item=SimpleNamespace(name="test_success", arguments='{"value":1}'),
        ),
    )
    output = RunItemStreamEvent(
        name="tool_output",
        item=SimpleNamespace(call_id="call-1", output='{"status":"error"}'),
    )

    assert runtime._convert_stream_event(called, collector) == {
        "type": "tool_start",
        "tool": "test_success",
        "tool_call_id": "call-1",
        "args": '{"value":1}',
    }
    assert runtime._convert_stream_event(output, collector) == {
        "type": "tool_result",
        "tool": "test_success",
        "tool_call_id": "call-1",
        "tool_result": '{"status":"error"}',
        "tool_status": "FAILED",
    }
    assert collector.calls() == [CapturedToolCall(
        tool_call_id="call-1",
        tool_name="test_success",
        arguments='{"value":1}',
    )]


class _JsonResponse:
    def __init__(self, body: dict) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.body


class _TextResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _GetOnlyClient:
    def __init__(self, responses: list) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict | None]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def get(self, url: str, params: dict | None = None):
        self.calls.append((url, params))
        return self.responses.pop(0)
