import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from agents import Agent, Runner, function_tool
from agents.items import ModelResponse, TResponseOutputItem, TResponseStreamEvent
from agents.models.interface import Model
from agents.stream_events import RunItemStreamEvent
from agents.tool import Tool
from agents.tool_context import ToolContext
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from agent_runner.context.builder import CapturedMessage
from agent_runner.runtime.cancellation import CancellationToken
from agent_runner.runtime.model_events import ModelToolCompleted, ModelToolStarted
from agent_runner.runtime.openai_agents_sdk_adapter import OpenAIAgentsSdkAdapter
from agent_runner.runtime.tool_loop import CapturedToolCall, ToolExecutionCollector
from agent_runner.tools.internal.catalog import build_internal_tool_registry
from agent_runner.tools.internal.web_search import (
    _DuckDuckGoResultParser,
    _SearchResult,
    _VisibleTextParser,
    _WebSearchClient,
)
from agent_runner.tools.registry import ToolDefinition


def _delay_definition(
    tool_key: str,
    delay: float,
    fails: bool = False,
    timeline: list[tuple[str, str, float]] | None = None,
) -> ToolDefinition:
    async def execute(value: int) -> dict[str, int]:
        """Run a delayed test Tool.

        Args:
            value: Value returned by the test Tool.
        """

        if timeline is not None:
            timeline.append((tool_key, "start", asyncio.get_running_loop().time()))

        try:
            await asyncio.sleep(delay)

            if fails:
                raise ValueError(f"failed:{value}")

            return {"value": value}
        finally:
            if timeline is not None:
                timeline.append((tool_key, "end", asyncio.get_running_loop().time()))

    sdk_tool = function_tool(
        name_override=tool_key.replace(".", "_"),
        failure_error_function=None,
    )(execute)

    return ToolDefinition.from_function_tool(tool_key, sdk_tool)


def _tool_context(tool_name: str, call_id: str, arguments: str) -> ToolContext[object]:
    return ToolContext(
        context=object(),
        tool_name=tool_name,
        tool_call_id=call_id,
        tool_arguments=arguments,
    )


async def _execute_builtin(tool_key: str, arguments: dict[str, object]) -> dict[str, Any]:
    definition = build_internal_tool_registry().get(tool_key)
    assert definition is not None
    assert definition.function_tool is not None
    arguments_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    result = await definition.function_tool.on_invoke_tool(
        _tool_context(definition.tool_name, "test-call", arguments_json),
        arguments_json,
    )

    return cast(dict[str, Any], result)


class _ToolLoopModel(Model):
    """Deterministic model that requests two Tools, then answers from their outputs."""

    def __init__(self, tool_names: tuple[str, str]) -> None:
        self.tool_names = tool_names
        self.inputs: list[object] = []

    async def get_response(self, *args: object, **kwargs: object) -> ModelResponse:
        model_input = kwargs["input"]
        self.inputs.append(model_input)

        if len(self.inputs) == 1:
            output: list[TResponseOutputItem] = [
                ResponseFunctionToolCall(
                    arguments='{"value":1}',
                    call_id="call-success",
                    name=self.tool_names[0],
                    type="function_call",
                ),
                ResponseFunctionToolCall(
                    arguments='{"value":2}',
                    call_id="call-failure",
                    name=self.tool_names[1],
                    type="function_call",
                ),
            ]
        else:
            output = [
                ResponseOutputMessage(
                    id="message-final",
                    content=[ResponseOutputText(annotations=[], text="Tool loop completed", type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ]

        return ModelResponse(output=output, usage=Usage(), response_id=f"response-{len(self.inputs)}")

    def stream_response(self, *args: object, **kwargs: object) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError


async def test_current_time_includes_timezone_offset_and_time() -> None:
    result = await _execute_builtin("builtin.current_time", {"timezone": "Asia/Shanghai"})

    assert result["timezone"] == "Asia/Shanghai"
    assert result["utc_offset"] == "+08:00"
    assert "+08:00" in result["local_datetime"]
    assert isinstance(result["unix_timestamp"], int)


async def test_current_time_rejects_unknown_timezone() -> None:
    with pytest.raises(ValueError, match="Unknown IANA timezone"):
        await _execute_builtin("builtin.current_time", {"timezone": "Invalid/AgentBreaker"})


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(12.5 + 7.5) / 4", 5.0),
        ("-3 * (+2 + 4)", -18),
    ],
)
async def test_calculator_supports_restricted_arithmetic_grammar(expression: str, expected: float) -> None:
    result = await _execute_builtin("builtin.calculator", {"expression": expression})
    assert result == {"expression": expression, "result": expected}


async def test_calculator_rejects_code_and_reports_division_by_zero() -> None:
    with pytest.raises(ValueError, match="unsupported syntax"):
        await _execute_builtin("builtin.calculator", {"expression": "pow(2, 8)"})
    with pytest.raises(ZeroDivisionError):
        await _execute_builtin("builtin.calculator", {"expression": "1 / 0"})


def test_builtin_definitions_are_derived_from_sdk_function_tools() -> None:
    registry = build_internal_tool_registry()
    definition = registry.get("builtin.calculator")

    assert definition is not None
    assert definition.function_tool is not None
    assert definition.tool_name == "calculate_expression"
    assert definition.strict is True
    assert definition.description.startswith("Evaluate an arithmetic expression")
    assert definition.parameters["properties"]["expression"]["description"].startswith("Code-style")


async def test_weather_uses_geocoding_and_current_weather(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _JsonResponse(
            {
                "results": [
                    {
                        "name": "Shanghai",
                        "admin1": "Shanghai",
                        "country": "China",
                        "latitude": 31.22,
                        "longitude": 121.46,
                    }
                ]
            }
        ),
        _JsonResponse(
            {
                "timezone": "Asia/Shanghai",
                "timezone_abbreviation": "GMT+8",
                "current": {"temperature_2m": 30.2},
                "current_units": {"temperature_2m": "C"},
            }
        ),
    ]
    client = _GetOnlyClient(responses)
    monkeypatch.setattr("agent_runner.tools.internal.weather.httpx.AsyncClient", lambda **_: client)

    result = await _execute_builtin("builtin.weather", {"location": " Shanghai "})

    assert result["location"]["name"] == "Shanghai"
    assert result["timezone"] == "Asia/Shanghai"
    assert result["current"]["temperature_2m"] == 30.2
    assert len(client.calls) == 2


async def test_web_search_parses_duckduckgo_and_keeps_partial_page_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_html = """
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">A title</a>
        <div class="result__snippet">A snippet</div>
        <a class="result__a" href="https://example.com/b">B title</a>
        <div class="result__snippet">B snippet</div>
    """
    client = _GetOnlyClient([_TextResponse(search_html)])
    monkeypatch.setattr("agent_runner.tools.internal.web_search.httpx.AsyncClient", lambda **_: client)

    async def fetch_result(_self: object, _client: object, result: _SearchResult) -> dict[str, object]:
        content = "Readable A" if result.title.strip() == "A title" else ""
        error = "" if content else "page failed"

        return {
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "content": content,
            "error": error,
        }

    monkeypatch.setattr(_WebSearchClient, "_fetch_result", fetch_result)

    result = await _execute_builtin("builtin.web_search", {"query": " AgentBreaker "})

    assert result["query"] == "AgentBreaker"
    assert len(result["results"]) == 2
    assert result["results"][0]["url"] == "https://example.com/a"
    assert result["results"][1]["error"] == "page failed"


def test_html_parsers_close_open_results_and_remove_hidden_content() -> None:
    search_parser = _DuckDuckGoResultParser(1)
    search_parser.feed('<a class="result__a" href="https://example.com">Title</a>')
    search_parser.close()
    assert search_parser.results == [_SearchResult(url="https://example.com", title="Title")]

    page_parser = _VisibleTextParser()
    page_parser.feed("<main>Visible<script>hidden()</script><style>hidden</style> text</main>")
    page_parser.close()
    assert page_parser.get_text() == "Visible\ntext"


async def test_sdk_runner_executes_parallel_tools_and_continues_with_outputs() -> None:
    timeline: list[tuple[str, str, float]] = []
    definitions = [
        _delay_definition("test.success", 0.05, timeline=timeline),
        _delay_definition("test.failure", 0.05, fails=True, timeline=timeline),
    ]
    collector = ToolExecutionCollector()
    adapter = OpenAIAgentsSdkAdapter(model_factory=SimpleNamespace())
    model = _ToolLoopModel((definitions[0].tool_name, definitions[1].tool_name))
    sdk_tools: list[Tool] = list(adapter._build_sdk_tools(definitions, collector, cancellation_token=None))
    sdk_agent = Agent(
        name="Tool loop integration",
        instructions="Run both Tools.",
        model=model,
        tools=sdk_tools,
    )

    result = await Runner.run(starting_agent=sdk_agent, input="Run the Tool loop.")

    starts = {tool_key: timestamp for tool_key, phase, timestamp in timeline if phase == "start"}
    ends = {tool_key: timestamp for tool_key, phase, timestamp in timeline if phase == "end"}
    assert max(starts.values()) < min(ends.values())
    assert result.final_output == "Tool loop completed"
    assert len(model.inputs) == 2

    second_input = model.inputs[1]
    assert isinstance(second_input, list)
    tool_outputs = {
        item["call_id"]: json.loads(item["output"])
        for item in second_input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    }
    assert tool_outputs == {
        "call-success": {"value": 1},
        "call-failure": {"status": "error", "error": "failed:2"},
    }
    success_execution = collector.get("call-success")
    failure_execution = collector.get("call-failure")
    assert success_execution is not None
    assert failure_execution is not None
    assert success_execution.status == "COMPLETED"
    assert failure_execution.status == "FAILED"


async def test_collector_returns_structured_failure_for_model_and_audit() -> None:
    definition = _delay_definition("test.failure", 0, fails=True)
    collector = ToolExecutionCollector()
    arguments = '{"value":2}'

    model_result = await collector.execute(
        tool_call_id="call-1",
        definition=definition,
        arguments_json=arguments,
        tool_context=_tool_context(definition.tool_name, "call-1", arguments),
        cancellation_token=None,
    )

    assert json.loads(model_result) == {"status": "error", "error": "failed:2"}
    execution = collector.get("call-1")
    assert execution is not None
    assert execution.status == "FAILED"
    assert execution.raw_result == model_result


async def test_cancelled_observed_call_builds_partial_turn_when_sdk_removes_raw_response() -> None:
    definition = _delay_definition("test.cancel", 10)
    collector = ToolExecutionCollector()
    collector.record_call(
        CapturedToolCall(
            tool_call_id="call-cancel",
            tool_name=definition.tool_name,
            arguments='{"value":1}',
        )
    )
    token = CancellationToken()
    task = asyncio.create_task(
        collector.execute(
            tool_call_id="call-cancel",
            definition=definition,
            arguments_json='{"value":1}',
            tool_context=_tool_context(definition.tool_name, "call-cancel", '{"value":1}'),
            cancellation_token=token,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    capture = OpenAIAgentsSdkAdapter(model_factory=SimpleNamespace())._build_observed_partial_capture(
        initial_messages=[CapturedMessage(role="user", content="cancel")],
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
    runtime = OpenAIAgentsSdkAdapter(model_factory=SimpleNamespace())
    collector = ToolExecutionCollector()
    cast(Any, collector)._executions["call-1"] = SimpleNamespace(tool_name="test_success", status="FAILED")
    called = RunItemStreamEvent(
        name="tool_called",
        item=cast(
            Any,
            SimpleNamespace(
                call_id="call-1",
                raw_item=SimpleNamespace(name="test_success", arguments='{"value":1}'),
            ),
        ),
    )
    output = RunItemStreamEvent(
        name="tool_output",
        item=cast(Any, SimpleNamespace(call_id="call-1", output='{"status":"error"}')),
    )

    assert runtime._convert_stream_event(called, collector) == ModelToolStarted(
        tool_name="test_success", tool_call_id="call-1", arguments_json='{"value":1}'
    )
    assert runtime._convert_stream_event(output, collector) == ModelToolCompleted(
        tool_name="test_success",
        tool_call_id="call-1",
        result='{"status":"error"}',
        status="FAILED",
    )
    assert collector.list_calls() == [
        CapturedToolCall(
            tool_call_id="call-1",
            tool_name="test_success",
            arguments='{"value":1}',
        )
    ]


class _JsonResponse:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.body


class _TextResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _GetOnlyClient:
    def __init__(self, responses: Sequence[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    async def __aenter__(self) -> "_GetOnlyClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    async def get(self, url: str, params: dict[str, object] | None = None) -> object:
        self.calls.append((url, params))

        return self.responses.pop(0)
