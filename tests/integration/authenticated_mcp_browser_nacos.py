"""Safely publish and restore isolated Nacos profiles for authenticated MCP browser tests."""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from v2.nacos import ConfigParam  # noqa: E402

from agent_runner.config import Settings  # noqa: E402
from agent_runner.nacos_config import NacosConfigLoader  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse one non-secret file-oriented Nacos configuration operation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("backup", "publish", "publish-fixture", "restore"))
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--required-secret", default="")
    parser.add_argument("--optional-secret", default="")
    parser.add_argument("--fixture-port", type=int, default=8766)
    parser.add_argument("--model-port", type=int, default=8767)
    return parser.parse_args()


def write_backup(path: Path, content: str) -> None:
    """Persist the authoritative snapshot without rendering it to stdout or stderr."""
    if not content.strip():
        raise RuntimeError("Nacos agent-runner configuration is empty")
    parsed = yaml.safe_load(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("Nacos agent-runner configuration must be a YAML object")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def merge_profile(base: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Replace only the test-owned model, MCP, and local-agent sections of one saved snapshot."""
    merged = dict(base)
    merged["lite_llm"] = profile["lite_llm"]
    merged["mcp"] = profile["mcp"]
    merged["local_agent_config"] = profile["local_agent_config"]
    return merged


def build_fixture_profile(
    required_secret: str,
    optional_secret: str,
    fixture_port: int,
    model_port: int,
) -> dict[str, Any]:
    """Build the complete local authenticated-MCP profile without exposing path or JSON editing to operators."""
    catalog = {
        "mcpServers": {
            "authenticated-required": {
                "url": f"http://127.0.0.1:{fixture_port}/mcp",
                "headers": {"Authorization": "Bearer ${secret:AUTHENTICATED_MCP_REQUIRED}"},
                "schema_cache_ttl_seconds": 0,
            },
            "authenticated-optional": {
                "url": f"http://127.0.0.1:{fixture_port}/mcp",
                "headers": {"Authorization": "Bearer ${secret:AUTHENTICATED_MCP_OPTIONAL}"},
                "schema_cache_ttl_seconds": 0,
            },
        }
    }
    return {
        "lite_llm": {
            "base_url": f"http://127.0.0.1:{model_port}/v1",
            "api_key": "authenticated-mcp-model-fixture",
            "request_timeout_seconds": 20,
            "max_retries": 0,
        },
        "mcp": {
            "catalog_json": json.dumps(catalog, separators=(",", ":")),
            "pool_max_connections_per_server": 2,
            "pool_idle_timeout_seconds": 30,
            "pool_borrow_timeout_seconds": 5,
            "secrets": {
                "AUTHENTICATED_MCP_REQUIRED": required_secret,
                "AUTHENTICATED_MCP_OPTIONAL": optional_secret,
            },
        },
        "local_agent_config": {
            "enabled": True,
            "path": "./tests/integration/authenticated_mcp_browser_agents.json",
        },
    }


async def run() -> None:
    """Execute a backup, test-profile publish, or exact restore through the Nacos SDK."""
    args = parse_arguments()
    loader = NacosConfigLoader.from_settings(Settings())
    await loader.initialize()
    try:
        client = loader.config_client
        if client is None:
            raise RuntimeError("Nacos configuration client is unavailable")
        parameter = ConfigParam(data_id=loader.data_id, group=loader.group, type="yaml")
        if args.operation == "backup":
            content = await client.get_config(parameter)
            write_backup(args.backup, content)
            print(json.dumps({"operation": "backup", "sha256": hashlib.sha256(content.encode()).hexdigest()}))
            return
        if args.operation == "restore":
            content = args.backup.read_text(encoding="utf-8")
        elif args.operation == "publish-fixture":
            if not args.required_secret and not args.optional_secret:
                raise RuntimeError("publish-fixture requires at least one explicit Secret value")
            base = yaml.safe_load(args.backup.read_text(encoding="utf-8"))
            if not isinstance(base, dict):
                raise RuntimeError("Nacos backup must be a YAML object")
            content = yaml.safe_dump(
                merge_profile(
                    base,
                    build_fixture_profile(
                        args.required_secret,
                        args.optional_secret,
                        args.fixture_port,
                        args.model_port,
                    ),
                ),
                allow_unicode=True,
                sort_keys=False,
            )
        else:
            if args.profile is None:
                raise RuntimeError("publish requires --profile")
            base = yaml.safe_load(args.backup.read_text(encoding="utf-8"))
            profile = json.loads(args.profile.read_text(encoding="utf-8"))
            if not isinstance(base, dict) or not isinstance(profile, dict):
                raise RuntimeError("Nacos backup and profile must be objects")
            content = yaml.safe_dump(merge_profile(base, profile), allow_unicode=True, sort_keys=False)
        if not await client.publish_config(
            ConfigParam(data_id=loader.data_id, group=loader.group, content=content, type="yaml")
        ):
            raise RuntimeError("Nacos configuration publish failed")
        print(json.dumps({"operation": args.operation, "sha256": hashlib.sha256(content.encode()).hexdigest()}))
    finally:
        await loader.close()


if __name__ == "__main__":
    asyncio.run(run())
