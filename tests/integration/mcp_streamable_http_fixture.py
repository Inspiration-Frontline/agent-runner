import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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
def search_current_information(query: str) -> dict[str, str]:
    """Return one current-information result for a natural-language research query."""

    return {
        "title": "Python 3.14 release information",
        "url": "https://www.python.org/downloads/",
        "query": query,
    }


@MCP.tool()
async def slow_current_information(query: str, delay_seconds: float = 4.0) -> dict[str, str]:
    """Return one current-information result after a deliberately slow remote lookup."""
    await asyncio.sleep(delay_seconds)

    return {
        "title": "Python 3.14 release information",
        "url": "https://www.python.org/downloads/",
        "query": query,
    }


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


@dataclass(frozen=True)
class StaticCredentialRequirement:
    """One fixture credential rule for either a Header or URL query parameter."""

    header_name: str = ""
    header_value: str = ""
    query_name: str = ""
    query_value: str = ""
    credential_file: str = ""
    rejection_status: int = 401


class StaticCredentialMiddleware:
    """Reject fixture requests whose configured static credential is missing or incorrect."""

    def __init__(self, app: ASGIApp, requirement: StaticCredentialRequirement) -> None:
        self._app = app
        self._requirement = requirement

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and not self._is_authorized(scope):
            await send(
                {
                    "type": "http.response.start",
                    "status": self._requirement.rejection_status,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"MCP credential rejected"})

            return
        await self._app(scope, receive, send)

    def _is_authorized(self, scope: Scope) -> bool:
        requirement = self._requirement
        expected_value = self._read_expected_value()

        if expected_value is None:
            return False

        if requirement.header_name:
            headers = dict(scope.get("headers", []))
            name = requirement.header_name.lower().encode("ascii")

            if headers.get(name) != expected_value.encode("utf-8"):
                return False

        if requirement.query_name:
            query = parse_qs(scope.get("query_string", b"").decode("utf-8"), keep_blank_values=True)

            if query.get(requirement.query_name) != [expected_value]:
                return False

        return True

    def _read_expected_value(self) -> str | None:
        requirement = self._requirement

        if not requirement.credential_file:
            return requirement.header_value or requirement.query_value

        try:
            return Path(requirement.credential_file).read_text(encoding="utf-8").strip()
        except OSError:
            return None


class CommitThenDropMiddleware:
    """Simulate uncertain delivery and expose credential-protected fixture evidence to tests."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("method") == "GET" and scope.get("path") == "/__test/ledger":
            body = json.dumps(LEDGER, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})

            return

        if scope["type"] != "http" or scope.get("method") != "POST":
            await self._app(scope, receive, send)

            return

        messages: list[Message] = []
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

        async def replay_receive() -> Message:
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
    parser.add_argument("--auth-header-name", default="")
    parser.add_argument("--auth-header-value", default="")
    parser.add_argument("--auth-query-name", default="")
    parser.add_argument("--auth-query-value", default="")
    parser.add_argument("--auth-credential-file", default="")
    parser.add_argument("--auth-failure-status", type=int, choices=(401, 403), default=401)
    parser.add_argument("--pid-file", default="")
    args = parser.parse_args()
    app: ASGIApp = MCP.streamable_http_app()
    app = CommitThenDropMiddleware(app)
    header_name = args.auth_header_name
    header_value = args.auth_header_value

    if args.bearer_token:
        if header_name or header_value:
            parser.error("--bearer-token cannot be combined with custom Header options")
        header_name = "Authorization"
        header_value = f"Bearer {args.bearer_token}"
    credential_file = args.auth_credential_file

    if header_name and bool(header_value) == bool(credential_file):
        parser.error("Header authentication requires exactly one static value or credential file")

    if args.auth_query_name and bool(args.auth_query_value) == bool(credential_file):
        parser.error("URL query authentication requires exactly one static value or credential file")

    if (header_value or credential_file) and not header_name and not args.auth_query_name:
        parser.error("an authentication target name is required")

    if args.auth_query_value and not args.auth_query_name:
        parser.error("URL query authentication requires a parameter name")

    if credential_file and bool(header_name) == bool(args.auth_query_name):
        parser.error("a credential file must authenticate exactly one Header or query parameter")

    if header_name or args.auth_query_name:
        app = StaticCredentialMiddleware(
            app,
            StaticCredentialRequirement(
                header_name=header_name,
                header_value=header_value,
                query_name=args.auth_query_name,
                query_value=args.auth_query_value,
                credential_file=credential_file,
                rejection_status=args.auth_failure_status,
            ),
        )

    if args.pid_file:
        Path(args.pid_file).write_text(str(os.getpid()), encoding="ascii")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
