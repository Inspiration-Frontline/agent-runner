"""
Agent configuration loader module.

This module provides functionality to load agent configurations from either
local JSON files or remote configuration center service, with Redis-based
caching for improved performance and configuration update propagation.
"""

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import redis.asyncio as aioredis

from config import AgentConfig, MemoryPolicy, PROJECT_ROOT, settings

logger = logging.getLogger(__name__)


class AgentConfigLoader:
    """
    Agent configuration loader with Redis caching support.

    This class handles loading agent configurations from multiple sources:
    1. Redis cache (first priority for performance)
    2. Local JSON configuration files (fallback for local development)
    3. Remote agent configuration center service (production source)

    The loader implements a caching strategy with TTL (time-to-live) to ensure
    configuration changes propagate within a reasonable timeframe while still
    benefiting from cache performance.

    Attributes:
        base_url: Base URL of the agent configuration center service.
        client: HTTP client for remote API calls.
        redis_client: Redis client for caching agent configurations.
        local_config_path: Path to local agent configuration JSON file.
        cache_ttl_seconds: TTL for cached configurations in Redis.
    """

    def __init__(self):
        """
        Initialize the agent configuration loader.

        Sets up HTTP client for remote calls, Redis client for caching,
        and resolves the local configuration file path.
        """
        self.base_url = settings.agent_config_center_url
        self.client = httpx.AsyncClient(timeout=30.0)

        # Initialize Redis client for caching
        self.redis_client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password if settings.redis_password else None,
            db=settings.redis_db,
            decode_responses=True,
        )
        self.cache_ttl_seconds = settings.agent_config_cache_ttl_seconds

        # Resolve local configuration path
        self.local_config_path = Path(settings.local_agent_config_path)
        if not self.local_config_path.is_absolute():
            self.local_config_path = PROJECT_ROOT / self.local_config_path

    async def load(self, agent_id: str, version: str | None = None) -> AgentConfig:
        """
        Load agent configuration from cache, local file, or remote service.

        The loading priority is:
        1. Check Redis cache for existing configuration
        2. If cache miss and local config enabled, try local JSON file
        3. If local config not found, fetch from remote configuration center
        4. Cache the result in Redis with TTL

        Args:
            agent_id: Unique identifier of the agent to load.
            version: Optional version string. If None, loads the latest version.

        Returns:
            AgentConfig: The loaded agent configuration object.

        Raises:
            ValueError: If the agent configuration cannot be found or loaded.
            httpx.HTTPError: If remote API call fails.
        """
        cache_key = f"agent_config:{agent_id}:{version or 'latest'}"

        # Try to load from Redis cache first
        cached_config = await self._get_from_cache(cache_key)
        if cached_config:
            logger.debug(f"Loaded agent config from cache: {cache_key}")
            return cached_config

        # Try local configuration if enabled
        if settings.local_agent_config_enabled:
            local_config = self._load_local_config(agent_id, version)
            if local_config:
                await self._set_cache(cache_key, local_config)
                logger.debug(f"Loaded agent config from local file: {cache_key}")
                return local_config

        # Fetch from remote configuration center
        try:
            url = f"{self.base_url}/api/v1/agents/{agent_id}"
            if version:
                url += f"?version={version}"

            response = await self.client.get(url)
            if response.status_code == 200:
                data = response.json()
                config = self._parse_config(data)
                await self._set_cache(cache_key, config)
                logger.debug(f"Loaded agent config from remote service: {cache_key}")
                return config

            raise ValueError(f"Failed to load agent config: {response.status_code}")

        except Exception as e:
            logger.exception(f"Error loading agent config for {agent_id}")
            raise

    async def _get_from_cache(self, cache_key: str) -> AgentConfig | None:
        """
        Retrieve agent configuration from Redis cache.

        Args:
            cache_key: The Redis key for the cached configuration.

        Returns:
            AgentConfig | None: The cached configuration if found, None otherwise.
        """
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return self._parse_config(data)
        except Exception as e:
            logger.warning(f"Failed to get config from Redis cache: {e}")
        return None

    async def _set_cache(self, cache_key: str, config: AgentConfig):
        """
        Store agent configuration in Redis cache with TTL.

        Args:
            cache_key: The Redis key for caching the configuration.
            config: The agent configuration to cache.
        """
        try:
            config_data = {
                "agent_id": config.agent_id,
                "version": config.version,
                "model": config.model,
                "system_prompt": config.system_prompt,
                "tools": config.tools,
                "mcp_servers": config.mcp_servers,
                "memory_policy": {
                    "profile": config.memory_policy.profile,
                    "rag": config.memory_policy.rag,
                },
                "max_output_tokens": config.max_output_tokens,
                "temperature": config.temperature,
                "mock_response": config.mock_response,
            }
            await self.redis_client.setex(
                cache_key, self.cache_ttl_seconds, json.dumps(config_data)
            )
        except Exception as e:
            logger.warning(f"Failed to set config in Redis cache: {e}")

    def _load_local_config(self, agent_id: str, version: str | None = None) -> AgentConfig | None:
        """
        Load agent configuration from local JSON file.

        Args:
            agent_id: Unique identifier of the agent to load.
            version: Optional version string. If None, matches any version.

        Returns:
            AgentConfig | None: The loaded configuration if found, None otherwise.
        """
        if not self.local_config_path.exists():
            return None

        with self.local_config_path.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)

        agents = data.get("agents", [])
        if isinstance(agents, dict):
            agents = list(agents.values())

        matching_agents = [
            agent
            for agent in agents
            if agent.get("agent_id") == agent_id and (version is None or agent.get("version") == version)
        ]
        if not matching_agents:
            return None

        return self._parse_config(matching_agents[0])

    def _parse_config(self, data: dict[str, Any]) -> AgentConfig:
        """
        Parse raw configuration data into AgentConfig object.

        Args:
            data: Raw dictionary containing agent configuration data.

        Returns:
            AgentConfig: The parsed agent configuration object.
        """
        memory_policy_data = data.get("memory_policy", {})
        memory_policy = MemoryPolicy(
            profile=memory_policy_data.get("profile", True),
            rag=memory_policy_data.get("rag", True),
        )

        return AgentConfig(
            agent_id=data["agent_id"],
            version=data.get("version", "latest"),
            model=data.get("model", "Qwen/Qwen3-4B"),
            system_prompt=data.get("system_prompt", ""),
            tools=data.get("tools", []),
            mcp_servers=data.get("mcp_servers", []),
            memory_policy=memory_policy,
            max_output_tokens=data.get("max_output_tokens", settings.max_output_tokens),
            temperature=data.get("temperature", 0.7),
            mock_response=data.get("mock_response"),
        )

    async def invalidate_cache(self, agent_id: str | None = None):
        """
        Invalidate cached agent configurations in Redis.

        This method clears cached configurations to force fresh loads on next request.
        It can clear all caches or only caches for a specific agent.

        Args:
            agent_id: Optional agent ID. If provided, only clears caches for that agent.
                     If None, clears all agent configuration caches.
        """
        try:
            if agent_id:
                # Find and delete all cache keys for this agent
                pattern = f"agent_config:{agent_id}:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    logger.info(f"Invalidated cache for agent {agent_id}: {len(keys)} keys")
            else:
                # Clear all agent configuration caches
                pattern = "agent_config:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    logger.info(f"Invalidated all agent config caches: {len(keys)} keys")
        except Exception as e:
            logger.warning(f"Failed to invalidate cache: {e}")

    async def close(self):
        """
        Close HTTP and Redis client connections.

        This method should be called during application shutdown to properly
        release resources and connections.
        """
        await self.client.aclose()
        await self.redis_client.close()
