import argparse
import asyncio
import json
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.types import ASGIApp, Receive, Scope, Send

LEDGER: list[dict[str, Any]] = []
MCP = FastMCP(
    "agentbreaker-mcp-test-fixture",
    host="127.0.0.1",
    port=8765,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)


@MCP.tool()
def echo(value: str) -> dict[str, Any]:
    return {"value": value, "ledger_size": len(LEDGER)}


@MCP.tool()
def write_record(key: str, value: str) -> dict[str, Any]:
    record = {"operation": "write", "key": key, "value": value}
    LEDGER.append(record)
    return record


@MCP.tool()
def delete_record(key: str) -> dict[str, Any]:
    record = {"operation": "delete", "key": key}
    LEDGER.append(record)
    return record


@MCP.tool()
def external_communication(destination: str, message: str) -> dict[str, Any]:
    record = {"operation": "external_communication", "destination": destination, "message": message}
    LEDGER.append(record)
    return record


@MCP.tool()
def explicit_failure(message: str = "fixture failure") -> None:
    raise RuntimeError(message)


@MCP.tool()
async def slow_pre_commit(key: str, delay_seconds: float = 1.0) -> dict[str, Any]:
    await asyncio.sleep(delay_seconds)
    record = {"operation": "slow_pre_commit", "key": key}
    LEDGER.append(record)
    return record


@MCP.tool()
def commit_then_drop(key: str) -> dict[str, Any]:
    return {"operation": "commit_then_drop", "key": key}


@MCP.tool()
async def parallel_probe(probe_id: str, delay_seconds: float = 0.1) -> dict[str, Any]:
    await asyncio.sleep(delay_seconds)
    record = {"operation": "parallel_probe", "probe_id": probe_id}
    LEDGER.append(record)
    return record


@MCP.tool()
def read_ledger() -> list[dict[str, Any]]:
    return list(LEDGER)


@MCP.tool()
def clear_ledger() -> dict[str, int]:
    count = len(LEDGER)
    LEDGER.clear()
    return {"cleared": count}


class StaticBearerMiddleware:
    def __init__(self, app: ASGIApp, token: str) -> None:
        self._app = app
        self._expected = f"Bearer {token}".encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            if headers.get(b"authorization") != self._expected:
                await send({"type": "http.response.start", "status": 401, "headers": []})
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
        await self._app(scope, receive, send)


class CommitThenDropMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self._app(scope, receive, send)
            return

        messages: list[dict[str, Any]] = []
        body = bytearray()
        while True:
            message = await receive()
            messages.append(message)
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}
        params = payload.get("params", {}) if isinstance(payload, dict) else {}
        if payload.get("method") == "tools/call" and params.get("name") == "commit_then_drop":
            arguments = params.get("arguments", {})
            LEDGER.append({"operation": "commit_then_drop", "key": str(arguments.get("key", ""))})
            raise ConnectionResetError("fixture dropped the response after committing the side effect")

        message_index = 0

        async def replay_receive() -> dict[str, Any]:
            nonlocal message_index
            if message_index < len(messages):
                message = messages[message_index]
                message_index += 1
                return message
            return {"type": "http.disconnect"}

        await self._app(scope, replay_receive, send)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bearer-token", default="")
    args = parser.parse_args()
    app: ASGIApp = MCP.streamable_http_app()
    app = CommitThenDropMiddleware(app)
    if args.bearer_token:
        app = StaticBearerMiddleware(app, args.bearer_token)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
