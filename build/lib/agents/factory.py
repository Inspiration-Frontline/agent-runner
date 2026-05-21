import logging
from typing import Any

from agents.config_models import AgentDefinition, MemoryPolicy
from config import AgentConfig

logger = logging.getLogger(__name__)


class AgentFactory:
    def __init__(self):
        self._agent_registry: dict[str, Any] = {}

    async def create(self, config: AgentConfig) -> AgentDefinition:
        agent = AgentDefinition(
            agent_id=config.agent_id,
            version=config.version,
            name=config.agent_id,
            description=f"Agent {config.agent_id}",
            model=config.model,
            system_prompt=config.system_prompt,
            tools=config.tools,
            mcp_servers=config.mcp_servers,
            memory_policy=MemoryPolicy(
                profile=config.memory_policy.profile,
                rag=config.memory_policy.rag,
            ),
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
        )

        self._agent_registry[config.agent_id] = agent
        logger.info(f"Created agent: {config.agent_id} v{config.version}")

        return agent

    async def create_with_overrides(
        self,
        config: AgentConfig,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AgentDefinition:
        agent = await self.create(config)

        if model:
            agent.model = model
        if temperature is not None:
            agent.temperature = temperature
        if max_output_tokens is not None:
            agent.max_output_tokens = max_output_tokens

        return agent

    def get_agent(self, agent_id: str) -> AgentDefinition | None:
        return self._agent_registry.get(agent_id)

    def remove_agent(self, agent_id: str):
        if agent_id in self._agent_registry:
            del self._agent_registry[agent_id]
