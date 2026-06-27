import logging
from typing import Any

from agent_definitions.config_models import AgentDefinition, MemoryPolicy
from config import AgentConfig

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory for creating and managing agent instances.

    This factory handles the creation of AgentDefinition objects from
    AgentConfig instances, maintains an in-memory registry of created agents,
    and supports runtime overrides of agent parameters.

    Attributes:
        _agent_registry: In-memory registry mapping agent_id to AgentDefinition instances.
    """

    def __init__(self):
        """
        Initialize the agent factory with an empty registry.
        """
        self._agent_registry: dict[str, Any] = {}

    async def create(self, config: AgentConfig) -> AgentDefinition:
        """
        Create an agent instance from configuration.

        Args:
            config: Agent configuration containing all required parameters.

        Returns:
            AgentDefinition: The created agent instance.

        Note:
            The created agent is automatically registered in the internal registry.
        """
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
        """
        Create an agent instance with runtime parameter overrides.

        Args:
            config: Base agent configuration.
            model: Optional override for the model identifier.
            temperature: Optional override for the temperature parameter.
            max_output_tokens: Optional override for max output tokens.

        Returns:
            AgentDefinition: The created agent instance with overrides applied.
        """
        agent = await self.create(config)

        if model:
            agent.model = model
        if temperature is not None:
            agent.temperature = temperature
        if max_output_tokens is not None:
            agent.max_output_tokens = max_output_tokens

        return agent

    def get_agent(self, agent_id: str) -> AgentDefinition | None:
        """
        Retrieve a previously created agent from the registry.

        Args:
            agent_id: The unique identifier of the agent to retrieve.

        Returns:
            AgentDefinition | None: The agent instance if found, None otherwise.
        """
        return self._agent_registry.get(agent_id)

    def remove_agent(self, agent_id: str):
        """
        Remove an agent from the registry.

        Args:
            agent_id: The unique identifier of the agent to remove.
        """
        if agent_id in self._agent_registry:
            del self._agent_registry[agent_id]
