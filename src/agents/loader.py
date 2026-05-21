import logging
import json
from pathlib import Path
from typing import Any

import httpx

from config import AgentConfig, MemoryPolicy, PROJECT_ROOT, settings

logger = logging.getLogger(__name__)


class AgentConfigLoader:
    def __init__(self):
        self.base_url = settings.agent_config_center_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self._cache: dict[str, AgentConfig] = {}
        self.local_config_path = Path(settings.local_agent_config_path)
        if not self.local_config_path.is_absolute():
            self.local_config_path = PROJECT_ROOT / self.local_config_path

    async def load(self, agent_id: str, version: str | None = None) -> AgentConfig:
        cache_key = f"{agent_id}:{version or 'latest'}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        if settings.local_agent_config_enabled:
            local_config = self._load_local_config(agent_id, version)
            if local_config:
                self._cache[cache_key] = local_config
                return local_config

        try:
            url = f"{self.base_url}/api/v1/agents/{agent_id}"
            if version:
                url += f"?version={version}"

            response = await self.client.get(url)
            if response.status_code == 200:
                data = response.json()
                config = self._parse_config(data)
                self._cache[cache_key] = config
                return config

            raise ValueError(f"Failed to load agent config: {response.status_code}")

        except Exception as e:
            logger.exception(f"Error loading agent config for {agent_id}")
            raise

    def _load_local_config(self, agent_id: str, version: str | None = None) -> AgentConfig | None:
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

    def invalidate_cache(self, agent_id: str | None = None):
        if agent_id:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{agent_id}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()

    async def close(self):
        await self.client.aclose()
