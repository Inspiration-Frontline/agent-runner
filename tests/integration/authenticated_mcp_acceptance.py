"""Local official-SDK acceptance for authenticated Streamable HTTP MCP Servers."""

import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent_runner.agent_definitions.config_models import MCPServerBinding  # noqa: E402
from agent_runner.config import Settings  # noqa: E402
from agent_runner.mcps.catalog import McpServerCatalog  # noqa: E402
from agent_runner.mcps.connection_pool import McpConnectionPool  # noqa: E402
from agent_runner.mcps.failures import McpFailureCode  # noqa: E402
from agent_runner.mcps.sdk_runtime import (  # noqa: E402
    DurableMcpServer,
    RequiredMcpServerUnavailableError,
    SdkMcpRuntime,
)
from agent_runner.mcps.secrets import McpSecretSnapshot  # noqa: E402
from agent_runner.observability.logging import setup_logging  # noqa: E402


class MutableSecrets:
    """Acceptance-only atomic snapshot supplier used to publish credential rotations."""

    def __init__(self, values: dict[str, str], revision: int = 1) -> None:
        self.snapshot = McpSecretSnapshot.create(values, revision, use_environment_fallback=False)

    def get_snapshot(self) -> McpSecretSnapshot:
        return self.snapshot

    def replace(self, values: dict[str, str], revision: int) -> None:
        self.snapshot = McpSecretSnapshot.create(values, revision, use_environment_fallback=False)


class EvidenceRecorder:
    """In-memory equivalent of the durable dispatch recorder for local acceptance assertions."""

    def __init__(self) -> None:
        self.states: list[str] = []

    async def before_dispatch(
        self,
        attempt_id: str,
        tool_call_id: str,
        turn_number: int,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        self.states.append("DISPATCHING")

    async def after_dispatch(self, attempt_id: str, state: str, recovery_reason: str = "") -> None:
        self.states.append(state)


@dataclass
class FixtureProcess:
    """One owned fixture process and its temporary credential/log artifacts."""

    process: subprocess.Popen[str]
    port: int
    credential_file: Path
    stdout_path: Path
    stderr_path: Path
    pid_path: Path

    def stop(self) -> None:
        """Terminate only this fixture's process tree, then wait for every child to exit."""

        if self.pid_path.exists():
            server_pid = int(self.pid_path.read_text(encoding="ascii"))
            with contextlib.suppress(OSError):
                os.kill(server_pid, signal.SIGTERM)

        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()

            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)

    def assert_logs_redacted(self, secrets: tuple[str, ...]) -> None:
        """Reject any synthetic credential found in fixture stdout or stderr."""
        logs = self.stdout_path.read_text(encoding="utf-8") + self.stderr_path.read_text(encoding="utf-8")

        for secret in secrets:
            if secret in logs:
                raise AssertionError("Fixture logs exposed a synthetic credential")


def reserve_port() -> int:
    """Reserve an ephemeral loopback port long enough to select a fixture endpoint."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))

        return int(listener.getsockname()[1])


def start_fixture(
    directory: Path,
    expected_credential: str,
    rejection_status: int,
    header_name: str = "",
    query_name: str = "",
) -> FixtureProcess:
    """Start one authenticated fixture without placing its credential on the command line."""
    port = reserve_port()
    credential_file = directory / f"credential-{port}.txt"
    credential_file.write_text(expected_credential, encoding="utf-8")
    stdout_path = directory / f"fixture-{port}.out.log"
    stderr_path = directory / f"fixture-{port}.err.log"
    pid_path = directory / f"fixture-{port}.pid"
    args = [
        sys.executable,
        str(Path(__file__).with_name("mcp_streamable_http_fixture.py")),
        "--port",
        str(port),
        "--auth-credential-file",
        str(credential_file),
        "--auth-failure-status",
        str(rejection_status),
        "--pid-file",
        str(pid_path),
    ]

    if header_name:
        args.extend(["--auth-header-name", header_name])
    else:
        args.extend(["--auth-query-name", query_name])
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            args,
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    fixture = FixtureProcess(process, port, credential_file, stdout_path, stderr_path, pid_path)
    wait_for_fixture(fixture, expected_credential, header_name, query_name)

    return fixture


def wait_for_fixture(
    fixture: FixtureProcess,
    expected_credential: str,
    header_name: str,
    query_name: str,
) -> None:
    """Wait until authenticated traffic reaches the FastMCP application boundary."""
    deadline = time.monotonic() + 20

    while time.monotonic() < deadline:
        if fixture.process.poll() is not None:
            raise RuntimeError("Authenticated MCP fixture exited during startup")
        headers = {header_name: expected_credential} if header_name else {}
        params = {query_name: expected_credential} if query_name else {}
        try:
            response = httpx.get(
                f"http://127.0.0.1:{fixture.port}/mcp",
                headers=headers,
                params=params,
                timeout=0.5,
            )

            if response.status_code not in {401, 403}:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)

    raise TimeoutError("Authenticated MCP fixture startup timed out")


def catalog_json(port: int, form: str) -> str:
    """Build an unresolved one-server Catalog document for one authentication form."""
    profile: dict[str, Any] = {"url": f"http://127.0.0.1:{port}/mcp", "schema_cache_ttl_seconds": 0}
    if form == "bearer":
        profile["headers"] = {"Authorization": "Bearer ${secret:FIXTURE_KEY}"}
    elif form == "custom":
        profile["headers"] = {"x-api-key": "${secret:FIXTURE_KEY}"}
    elif form == "url":
        profile["url"] += "?api_key=${secret:FIXTURE_KEY}"
        profile["allow_url_secret"] = True
    else:
        raise ValueError("Unsupported fixture authentication form")

    return json.dumps({"mcpServers": {"fixture": profile}}, separators=(",", ":"))


async def call_tool(
    runtime: SdkMcpRuntime,
    recorder: EvidenceRecorder,
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str,
) -> Any:
    """Call one fixture Tool through the production request-scoped SDK adapter."""
    async with runtime.session([MCPServerBinding("fixture", required=True)], recorder) as session:
        server = session.servers[0]

        if not isinstance(server, DurableMcpServer):
            raise TypeError("Acceptance session did not produce the durable MCP adapter")

        return await server.call_tool(
            tool_name,
            arguments,
            {"agentbreaker/tool_call_id": call_id, "agentbreaker/turn_number": 1},
        )


async def verify_static_form(
    directory: Path,
    form: str,
    correct_value: str,
    rejection_status: int,
) -> dict[str, Any]:
    """Verify success, missing, wrong, required, optional, and concurrent behavior for one form."""
    expected = f"Bearer {correct_value}" if form == "bearer" else correct_value
    header_name = "Authorization" if form == "bearer" else "x-api-key" if form == "custom" else ""
    query_name = "api_key" if form == "url" else ""
    fixture = start_fixture(directory, expected, rejection_status, header_name, query_name)
    pool = McpConnectionPool()
    secrets = MutableSecrets({"FIXTURE_KEY": correct_value})
    runtime = SdkMcpRuntime(
        McpServerCatalog.from_json(catalog_json(fixture.port, form), secrets),
        connection_pool=pool,
        settings=Settings(),
    )

    try:
        recorder = EvidenceRecorder()
        result = await call_tool(runtime, recorder, "echo", {"value": form}, f"{form}-success")

        if bool(getattr(result, "isError", False)) or recorder.states != ["DISPATCHING", "COMPLETED"]:
            raise AssertionError(f"{form} authenticated Tool call did not complete")

        secrets.replace({"FIXTURE_KEY": f"wrong-{form}"}, 2)
        expected_code = (
            McpFailureCode.AUTHENTICATION_REJECTED if rejection_status == 401 else McpFailureCode.AUTHORIZATION_DENIED
        )

        try:
            async with runtime.session([MCPServerBinding("fixture", required=True)]):
                raise AssertionError("Wrong required credential yielded an active session")
        except RequiredMcpServerUnavailableError as error:
            if error.diagnostics[0].failure_code != expected_code:
                raise AssertionError("Wrong credential received an unexpected failure classification") from error

        async with runtime.session([MCPServerBinding("fixture", required=False)]) as optional_session:
            if optional_session.servers or optional_session.diagnostics[0].failure_code != expected_code:
                raise AssertionError("Optional wrong credential did not degrade with a typed diagnostic")

        secrets.replace({}, 3)

        try:
            async with runtime.session([MCPServerBinding("fixture", required=True)]):
                raise AssertionError("Missing required credential yielded an active session")
        except RequiredMcpServerUnavailableError as error:
            if error.diagnostics[0].failure_code != McpFailureCode.SECRET_MISSING:
                raise AssertionError("Missing credential received an unexpected failure classification") from error

        secrets.replace({"FIXTURE_KEY": correct_value}, 4)
        concurrent_recorder = EvidenceRecorder()
        async with runtime.session([MCPServerBinding("fixture", required=True)], concurrent_recorder) as session:
            server = session.servers[0]

            if not isinstance(server, DurableMcpServer):
                raise TypeError("Concurrent acceptance session did not produce a durable adapter")
            results = await asyncio.gather(
                server.call_tool(
                    "parallel_probe",
                    {"probe_id": "one", "delay_seconds": 0.05},
                    {"agentbreaker/tool_call_id": f"{form}-parallel-1", "agentbreaker/turn_number": 1},
                ),
                server.call_tool(
                    "parallel_probe",
                    {"probe_id": "two", "delay_seconds": 0.05},
                    {"agentbreaker/tool_call_id": f"{form}-parallel-2", "agentbreaker/turn_number": 1},
                ),
            )

        if any(bool(getattr(item, "isError", False)) for item in results):
            raise AssertionError("Concurrent authenticated Tool calls failed")

        return {"form": form, "success": True, "rejection_status": rejection_status}
    finally:
        await pool.close()
        fixture.stop()
        fixture.assert_logs_redacted((correct_value, f"wrong-{form}"))


async def verify_in_flight_rotation(directory: Path) -> dict[str, Any]:
    """Prove a borrowed old connection finishes while new requests use the rotated credential."""
    old_secret = "rotation-old-secret"
    new_secret = "rotation-new-secret"
    fixture = start_fixture(directory, f"Bearer {old_secret}", 401, "Authorization", "")
    pool = McpConnectionPool()
    secrets = MutableSecrets({"FIXTURE_KEY": old_secret}, 1)
    runtime = SdkMcpRuntime(
        McpServerCatalog.from_json(catalog_json(fixture.port, "bearer"), secrets),
        connection_pool=pool,
        settings=Settings(),
    )
    old_recorder = EvidenceRecorder()
    new_recorder = EvidenceRecorder()

    try:
        async with runtime.session([MCPServerBinding("fixture", required=True)], old_recorder) as old_session:
            old_server = old_session.servers[0]

            if not isinstance(old_server, DurableMcpServer):
                raise TypeError("Rotation session did not produce a durable adapter")
            old_call = asyncio.create_task(
                old_server.call_tool(
                    "slow_pre_commit",
                    {"key": "in-flight", "delay_seconds": 0.5},
                    {"agentbreaker/tool_call_id": "rotation-old", "agentbreaker/turn_number": 1},
                )
            )
            await asyncio.sleep(0.1)
            fixture.credential_file.write_text(f"Bearer {new_secret}", encoding="utf-8")
            secrets.replace({"FIXTURE_KEY": new_secret}, 2)
            new_result = await call_tool(runtime, new_recorder, "echo", {"value": "rotated"}, "rotation-new")
            old_result = await old_call

        if bool(getattr(old_result, "isError", False)) or bool(getattr(new_result, "isError", False)):
            raise AssertionError("Credential rotation interrupted a Tool call")

        if old_recorder.states != ["DISPATCHING", "COMPLETED"]:
            raise AssertionError("Old in-flight call did not preserve terminal evidence")

        if new_recorder.states != ["DISPATCHING", "COMPLETED"]:
            raise AssertionError("Rotated request did not use the new credential")

        return {"rotation": True, "old_in_flight_completed": True, "new_request_completed": True}
    finally:
        await pool.close()
        fixture.stop()
        fixture.assert_logs_redacted((old_secret, new_secret))


async def main() -> None:
    """Run the complete isolated acceptance and print only non-secret pass evidence."""
    setup_logging()
    with tempfile.TemporaryDirectory(prefix="agentbreaker-mcp-auth-") as temporary_directory:
        directory = Path(temporary_directory)
        results = [
            await verify_static_form(directory, "bearer", "bearer-fixture-secret", 401),
            await verify_static_form(directory, "custom", "custom-fixture-secret", 403),
            await verify_static_form(directory, "url", "url-fixture-secret", 401),
            await verify_in_flight_rotation(directory),
        ]
    print(json.dumps(results, separators=(",", ":")))


if __name__ == "__main__":
    asyncio.run(main())
