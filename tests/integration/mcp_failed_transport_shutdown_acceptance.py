"""Verify failed Streamable HTTP transports leave no task-affine cleanup behind."""

import asyncio
import json
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

from authenticated_mcp_acceptance import MutableSecrets, start_fixture

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent_runner.agent_definitions.config_models import MCPServerBinding  # noqa: E402
from agent_runner.config import Settings  # noqa: E402
from agent_runner.mcps.catalog import McpServerCatalog  # noqa: E402
from agent_runner.mcps.connection_pool import McpConnectionPool  # noqa: E402
from agent_runner.mcps.sdk_runtime import DurableMcpServer, SdkMcpRuntime  # noqa: E402


async def main() -> None:
    """Invoke the commit/drop fixture, then close its failed pooled transport."""
    loop_errors: list[dict[str, object]] = []
    asyncio.get_running_loop().set_exception_handler(lambda _loop, context: loop_errors.append(context))
    with tempfile.TemporaryDirectory(prefix="agentbreaker-mcp-failed-shutdown-") as directory:
        secret = "failed-transport-shutdown-secret"
        fixture = start_fixture(Path(directory), f"Bearer {secret}", 401, "Authorization", "")
        pool = McpConnectionPool()
        secrets = MutableSecrets({"FIXTURE_KEY": secret})
        catalog = McpServerCatalog.from_json(
            json.dumps(
                {
                    "mcpServers": {
                        "fixture": {
                            "url": f"http://127.0.0.1:{fixture.port}/mcp",
                            "headers": {"Authorization": "Bearer ${secret:FIXTURE_KEY}"},
                        }
                    }
                }
            ),
            secrets,
        )
        runtime = SdkMcpRuntime(catalog, connection_pool=pool, settings=Settings())
        connection = None

        try:
            async with runtime.session([MCPServerBinding("fixture", required=True)]) as session:
                server = session.servers[0]

                if not isinstance(server, DurableMcpServer):
                    raise TypeError("Fixture did not produce a durable MCP server")
                connection = server._connection
                with suppress(BaseException):
                    await server.call_tool(
                        "commit_then_drop",
                        {"key": "shutdown-repro"},
                        {"agentbreaker/tool_call_id": "shutdown-repro", "agentbreaker/turn_number": 1},
                    )
            await pool.close()
            await asyncio.sleep(0)
        finally:
            fixture.stop()
            fixture.assert_logs_redacted((secret,))

        if connection is None:
            raise AssertionError("Fixture connection was not created")

        if connection.server.exit_stack._exit_callbacks:
            raise AssertionError("MCP server retained async exit callbacks after cleanup")

        if loop_errors:
            raise AssertionError(f"Event loop received {len(loop_errors)} cleanup error(s)")


if __name__ == "__main__":
    asyncio.run(main())
