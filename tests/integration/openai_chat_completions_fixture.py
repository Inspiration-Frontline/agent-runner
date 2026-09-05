"""Deterministic OpenAI-compatible streaming model used by full-browser tests."""

import argparse
import json
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

APP = FastAPI()


def _text_content(content: Any) -> str:
    """Flatten the OpenAI message content shapes needed by the fixture."""

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    return " ".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
    )


def _active_turn_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return messages at and after the latest user instruction."""
    messages = [message for message in payload.get("messages", []) if isinstance(message, dict)]
    latest_user_index = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default=0,
    )

    return messages[latest_user_index:]


def _requested_tool(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Select the named fixture capability from the latest user instruction."""
    active_messages = _active_turn_messages(payload)
    user_text = " ".join(
        _text_content(message.get("content")) for message in active_messages if message.get("role") == "user"
    ).lower()
    requested_suffix = "search_current_information"
    arguments: dict[str, Any] = {"query": "current Python release information"}
    if "deliberately slow" in user_text:
        requested_suffix = "slow_current_information"
        arguments["delay_seconds"] = 4.0
    elif "uncertain external delivery" in user_text:
        requested_suffix = "commit_then_drop"
        arguments = {"key": "authenticated-mcp-browser-once"}

    tools = payload.get("tools", [])
    names = [
        str(tool.get("function", {}).get("name", ""))
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
    matching_name = next((name for name in names if name.endswith(requested_suffix)), "")

    if not matching_name:
        raise ValueError(f"requested fixture Tool is absent: {requested_suffix}")

    return matching_name, arguments


def _has_tool_result(payload: dict[str, Any]) -> bool:
    """Return whether the latest user turn already contains a Tool result."""

    return any(message.get("role") == "tool" for message in _active_turn_messages(payload))


def _final_answer(payload: dict[str, Any]) -> str:
    """Explain uncertain delivery explicitly; otherwise return stable fixture content."""
    active_messages = _active_turn_messages(payload)
    user_text = " ".join(
        _text_content(message.get("content")) for message in active_messages if message.get("role") == "user"
    ).lower()

    if "uncertain external delivery" in user_text:
        return "The external action may have completed, but its outcome could not be confirmed. It was not retried."

    return "Python 3.14 release information: https://www.python.org/downloads/"


def _chunk(completion_id: str, model: str, delta: dict[str, Any], finish_reason: str | None) -> str:
    """Serialize one OpenAI Chat Completions SSE chunk."""
    body = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }

    return f"data: {json.dumps(body, separators=(',', ':'))}\n\n"


async def _stream_completion(payload: dict[str, Any]) -> AsyncIterator[str]:
    """Emit one deterministic Tool request or its concise final answer."""
    completion_id = f"chatcmpl-{uuid4().hex}"
    model = str(payload.get("model", "fixture-model"))
    yield _chunk(completion_id, model, {"role": "assistant"}, None)

    if _has_tool_result(payload):
        yield _chunk(
            completion_id,
            model,
            {"content": _final_answer(payload)},
            None,
        )
        yield _chunk(completion_id, model, {}, "stop")
    else:
        tool_name, arguments = _requested_tool(payload)
        yield _chunk(
            completion_id,
            model,
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call_{uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments, separators=(",", ":")),
                        },
                    }
                ]
            },
            None,
        )
        yield _chunk(completion_id, model, {}, "tool_calls")

    yield "data: [DONE]\n\n"


@APP.get("/health")
async def health() -> JSONResponse:
    """Expose a readiness probe for the browser-test controller."""

    return JSONResponse({"status": "healthy"})


@APP.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    """Serve the streaming subset of the OpenAI Chat Completions contract."""
    payload = await request.json()

    if not isinstance(payload, dict):
        return JSONResponse({"error": {"message": "request body must be an object"}}, status_code=400)

    if not payload.get("stream"):
        return JSONResponse({"error": {"message": "fixture requires stream=true"}}, status_code=400)

    try:
        stream = _stream_completion(payload)

        return StreamingResponse(stream, media_type="text/event-stream")
    except ValueError as error:
        return JSONResponse({"error": {"message": str(error)}}, status_code=400)


def main() -> None:
    """Run the deterministic model fixture on a caller-selected local port."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    uvicorn.run(APP, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
