import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

CASE_MESSAGES = {
    1201: "Use the available MCP tool to read the openai/openai-python wiki structure and return its first page.",
    1202: "Use the Microsoft Learn MCP search tool for Azure OpenAI Python quickstart and return one result title.",
    1203: "Use the Context7 MCP resolver to find the HTTPX library ID for async HTTP client usage.",
    1204: "Use the OpenAI Docs MCP listing tool with limit 3 and return the first URL.",
}


def wait_for_health(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Runner exited with code {process.returncode}")
        try:
            if httpx.get(f"{base_url}/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise TimeoutError("Runner health check timed out")


def run_case(
    project_root: Path,
    gateway_url: str,
    runner_url: str,
    output_dir: Path,
    agent_id: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({
        "SERVER_PORT": runner_url.rsplit(":", maxsplit=1)[1],
        "DEFAULT_AGENT_ID": str(agent_id),
        "LOCAL_AGENT_CONFIG_PATH": str(project_root / "tests" / "integration" / "phase12_agents.json"),
        "NACOS_ENABLED": "false",
        "OPEN_BROWSER_ON_STARTUP": "false",
    })
    stdout_path = output_dir / f"phase12-runner-{agent_id}.out.log"
    stderr_path = output_dir / f"phase12-runner-{agent_id}.err.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            [sys.executable, "src/main.py"],
            cwd=project_root,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        try:
            wait_for_health(runner_url, process)
            with httpx.Client(base_url=gateway_url, timeout=180) as client:
                created = client.post("/conversation/new", json={})
                created.raise_for_status()
                conversation_id = str(created.json()["data"]["conversationId"])
                with client.stream(
                    "POST",
                    "/v1/agent/chat/stream",
                    json={
                        "conversation_id": conversation_id,
                        "message": CASE_MESSAGES.get(
                            agent_id,
                            f"Run the configured Phase 12 case for Agent {agent_id}.",
                        ),
                        "file_ids": [],
                        "references": [],
                        "ui_locale": "en-US",
                    },
                ) as response:
                    response.raise_for_status()
                    sse = "\n".join(response.iter_lines())
            sse_path = output_dir / f"phase12-agent-{agent_id}.sse"
            sse_path.write_text(sse, encoding="utf-8")
            events = [
                json.loads(line.removeprefix("data: "))
                for line in sse.splitlines()
                if line.startswith("data: ")
            ]
            time.sleep(4)
            return {
                "agent_id": agent_id,
                "conversation_id": conversation_id,
                "event_types": [event["type"] for event in events],
                "tool_events": [
                    {
                        "type": event["type"],
                        "tool": event.get("tool"),
                        "tool_call_id": event.get("tool_call_id"),
                        "tool_status": event.get("tool_status"),
                    }
                    for event in events
                    if event["type"] in {"tool_start", "tool_result"}
                ],
                "terminal": events[-1]["type"] if events else "missing",
            }
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_ids", nargs="+", type=int)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8181")
    parser.add_argument("--runner-url", default="http://127.0.0.1:8101")
    parser.add_argument("--output-dir", default="../temp")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_case(project_root, args.gateway_url, args.runner_url, output_dir, agent_id)
        for agent_id in args.agent_ids
    ]
    manifest = output_dir / "phase12-live-acceptance.json"
    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
