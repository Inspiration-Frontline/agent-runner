"""
Test script for Nacos configuration integration.

This script tests:
1. Publish config to Nacos with different values than local defaults
2. Verify agent-runner reads Nacos values (priority test)
3. Update Nacos config
4. Verify agent-runner reads updated values (dynamic refresh test)
"""

import asyncio

import httpx
from v2.nacos import ConfigParam

from agent_runner.nacos_config import NacosConfigLoader

# Test configuration values (different from local defaults)
TEST_CONFIG_V1 = """
lite_llm:
  base_url: http://nacos-test-v1:5000
  api_key: sk-nacos-test-key-v1

services:
  agent_config_center_url: http://nacos-test-v1:9081
  conversation_service_url: http://nacos-test-v1:9082

redis:
  host: nacos-redis-v1
  port: 6380

cache:
  agent_config_ttl_seconds: 600
"""

TEST_CONFIG_V2 = """
lite_llm:
  base_url: http://nacos-test-v2:6000
  api_key: sk-nacos-test-key-v2

services:
  agent_config_center_url: http://nacos-test-v2:10081
  conversation_service_url: http://nacos-test-v2:10082

redis:
  host: nacos-redis-v2
  port: 6381

cache:
  agent_config_ttl_seconds: 900
"""

# Local default values (from config.py)
LOCAL_DEFAULTS = {
    "lite_llm_base_url": "http://localhost:4000",
    "agent_config_center_url": "http://localhost:8081",
    "redis_host": "localhost",
    "agent_config_cache_ttl_seconds": 300,
}


async def publish_config(loader: NacosConfigLoader, content: str):
    """Publish configuration to Nacos using the SDK's internal client."""
    if not loader.config_client:
        raise RuntimeError("Nacos client not initialized")

    published = await loader.config_client.publish_config(ConfigParam(
        data_id=loader.data_id,
        group=loader.group,
        content=content,
        type="yaml",
    ))
    if not published:
        raise RuntimeError(f"Failed to publish config to Nacos: data_id={loader.data_id}, group={loader.group}")
    print(f"Published config to Nacos: data_id={loader.data_id}, group={loader.group}")


async def wait_for_debug_config(expected: dict[str, object], timeout_seconds: float = 30.0) -> dict:
    """Poll the running service until the debug config endpoint returns expected values."""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_config = None
    async with httpx.AsyncClient() as client:
        while asyncio.get_running_loop().time() < deadline:
            resp = await client.get("http://localhost:8000/v1/agent/debug/config")
            resp.raise_for_status()
            last_config = resp.json()
            if all(last_config.get(key) == value for key, value in expected.items()):
                return last_config
            await asyncio.sleep(1)

    raise AssertionError(f"Expected config values {expected}, got {last_config}")


async def test_config_priority(loader: NacosConfigLoader):
    """Test that Nacos config overrides local defaults."""
    print("\n=== TEST 1: Config Priority (Nacos > Local) ===")

    # Publish V1 config (different from local defaults)
    await publish_config(loader, TEST_CONFIG_V1)

    config = await wait_for_debug_config(
        {
            "lite_llm_base_url": "http://nacos-test-v1:5000",
            "agent_config_center_url": "http://nacos-test-v1:9081",
            "redis_host": "nacos-redis-v1",
            "agent_config_cache_ttl_seconds": 600,
        }
    )

    print(f"Config from API: {config}")

    # Verify Nacos values override local defaults
    assert config["lite_llm_base_url"] == "http://nacos-test-v1:5000", \
        f"Expected Nacos value, got {config['lite_llm_base_url']}"
    assert config["agent_config_center_url"] == "http://nacos-test-v1:9081", \
        f"Expected Nacos value, got {config['agent_config_center_url']}"
    assert config["redis_host"] == "nacos-redis-v1", \
        f"Expected Nacos value, got {config['redis_host']}"
    assert config["agent_config_cache_ttl_seconds"] == 600, \
        f"Expected Nacos value, got {config['agent_config_cache_ttl_seconds']}"

    print("OK TEST 1 PASSED: Nacos config overrides local defaults")


async def test_dynamic_refresh(loader: NacosConfigLoader):
    """Test that config updates are reflected dynamically."""
    print("\n=== TEST 2: Dynamic Refresh ===")

    # Publish V2 config (updated values)
    await publish_config(loader, TEST_CONFIG_V2)

    config = await wait_for_debug_config(
        {
            "lite_llm_base_url": "http://nacos-test-v2:6000",
            "agent_config_center_url": "http://nacos-test-v2:10081",
            "redis_host": "nacos-redis-v2",
            "agent_config_cache_ttl_seconds": 900,
        }
    )

    print(f"Config from API: {config}")

    # Verify updated values
    assert config["lite_llm_base_url"] == "http://nacos-test-v2:6000", \
        f"Expected updated Nacos value, got {config['lite_llm_base_url']}"
    assert config["agent_config_center_url"] == "http://nacos-test-v2:10081", \
        f"Expected updated Nacos value, got {config['agent_config_center_url']}"
    assert config["redis_host"] == "nacos-redis-v2", \
        f"Expected updated Nacos value, got {config['redis_host']}"
    assert config["agent_config_cache_ttl_seconds"] == 900, \
        f"Expected updated Nacos value, got {config['agent_config_cache_ttl_seconds']}"

    print("OK TEST 2 PASSED: Dynamic refresh works correctly")


async def cleanup(loader: NacosConfigLoader):
    """Remove test config from Nacos."""
    print("\n=== CLEANUP ===")
    if loader.config_client:
        await loader.config_client.remove_config(ConfigParam(
            data_id=loader.data_id,
            group=loader.group,
        ))
        print("Removed test config from Nacos")
    await loader.close()


async def main():
    """Run all tests."""
    print("Starting Nacos configuration integration tests...")
    print("Prerequisites:")
    print("  1. Nacos server running at localhost:8848")
    print("  2. agent-runner service running at localhost:8000")
    print("  3. NACOS_ENABLED=true in environment")

    loader = NacosConfigLoader.from_env()
    try:
        await loader.initialize()
        await test_config_priority(loader)
        await test_dynamic_refresh(loader)
        print("\nOK ALL TESTS PASSED")
    except Exception as e:
        print(f"\nERROR TEST FAILED: {e}")
        raise
    finally:
        await cleanup(loader)


if __name__ == "__main__":
    asyncio.run(main())
