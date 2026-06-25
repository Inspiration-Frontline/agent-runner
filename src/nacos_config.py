"""
Nacos configuration loader module.

This module provides functionality to load configuration from Nacos configuration center,
with support for YAML format, configuration caching, and dynamic refresh through listeners.
"""

import asyncio
import logging
import os
from typing import Any

import yaml
from v2.nacos import ClientConfigBuilder, ConfigParam, GRPCConfig, NacosConfigService

logger = logging.getLogger(__name__)


class NacosConfigLoader:
    """
    Nacos configuration loader with async support.

    This class handles loading configuration from Nacos configuration center:
    - Supports YAML format configuration
    - Provides configuration caching and snapshot fallback
    - Supports dynamic configuration refresh through listeners
    - Falls back gracefully when Nacos is unavailable

    Attributes:
        config_client: Nacos configuration service client.
        data_id: Configuration Data ID in Nacos.
        group: Configuration group in Nacos.
        namespace: Nacos namespace for configuration isolation.
        enabled: Whether Nacos configuration is enabled.
    """

    def __init__(
        self,
        server_address: str = "127.0.0.1:8848",
        namespace: str = "agent-breaker-local",
        data_id: str = "agent-runner.yaml",
        group: str = "DEFAULT_GROUP",
        username: str = "nacos",
        password: str = "nacos",
        enabled: bool = True,
    ):
        """
        Initialize the Nacos configuration loader.

        Args:
            server_address: Nacos server address (default: 127.0.0.1:8848).
            namespace: Nacos namespace ID (default: agent-breaker-local).
            data_id: Configuration Data ID (default: agent-runner.yaml).
            group: Configuration group (default: DEFAULT_GROUP).
            username: Nacos authentication username (default: nacos).
            password: Nacos authentication password (default: nacos).
            enabled: Whether to enable Nacos configuration (default: True).
        """
        self.server_address = server_address
        self.namespace = namespace
        self.data_id = data_id
        self.group = group
        self.username = username
        self.password = password
        self.enabled = enabled
        self.config_client: NacosConfigService | None = None
        self._cached_config: dict[str, Any] = {}
        self._listener_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """
        Initialize the Nacos configuration client.

        Creates the NacosConfigService client and sets up configuration listener
        for dynamic refresh. This method should be called during application startup.
        """
        if not self.enabled:
            logger.info("Nacos configuration is disabled, using local configuration only")
            return

        try:
            client_config = (
                ClientConfigBuilder()
                .server_address(self.server_address)
                .namespace_id(self.namespace)
                .username(self.username)
                .password(self.password)
                .log_level("INFO")
                .grpc_config(GRPCConfig(grpc_timeout=5000))
                .build()
            )

            self.config_client = await NacosConfigService.create_config_service(client_config)
            logger.info(
                f"Nacos config client initialized: server={self.server_address}, "
                f"namespace={self.namespace}, data_id={self.data_id}, group={self.group}"
            )

            # Load initial configuration
            await self._load_and_cache_config()

            # Start configuration listener for dynamic refresh
            self._listener_task = asyncio.create_task(self._start_config_listener())

        except Exception as e:
            logger.warning(f"Failed to initialize Nacos client: {e}, falling back to local configuration")
            self.enabled = False

    async def _load_and_cache_config(self) -> dict[str, Any]:
        """
        Load configuration from Nacos and cache it.

        Returns:
            dict[str, Any]: The loaded configuration dictionary.
        """
        if not self.config_client:
            return {}

        try:
            config_param = ConfigParam(data_id=self.data_id, group=self.group)
            content = await self.config_client.get_config(config_param)

            if content:
                parsed_config = yaml.safe_load(content) or {}
                self._cached_config = parsed_config
                logger.debug(f"Loaded configuration from Nacos: {self.data_id}")
                return parsed_config

        except Exception as e:
            logger.warning(f"Failed to load config from Nacos: {e}")

        return {}

    async def _start_config_listener(self) -> None:
        """
        Start listening for configuration changes in Nacos.

        This method runs as a background task and updates the cached configuration
        whenever changes are detected in Nacos.
        """
        if not self.config_client:
            return

        async def config_listener(namespace_id: str, group: str, data_id: str, content: str) -> None:
            """
            Callback function for configuration changes.

            Args:
                namespace_id: The namespace ID of the configuration.
                group: The group of the configuration.
                data_id: The Data ID of the configuration.
                content: The new configuration content.
            """
            logger.info(f"Configuration changed in Nacos: data_id={data_id}, group={group}")
            if content:
                try:
                    parsed_config = yaml.safe_load(content) or {}
                    self._cached_config = parsed_config
                    logger.info(f"Configuration cache updated from Nacos")
                except Exception as e:
                    logger.warning(f"Failed to parse updated configuration: {e}")

        try:
            await self.config_client.add_listener(
                data_id=self.data_id,
                group=self.group,
                listener=config_listener,
            )
            logger.info(f"Started configuration listener for {self.data_id}")
        except Exception as e:
            logger.warning(f"Failed to add configuration listener: {e}")

    async def get_config(self) -> dict[str, Any]:
        """
        Get the current configuration from cache.

        Returns the cached configuration that was loaded from Nacos.
        If Nacos is disabled or unavailable, returns an empty dictionary.

        Returns:
            dict[str, Any]: The current configuration dictionary.
        """
        if not self.enabled:
            return {}

        return self._cached_config

    async def close(self) -> None:
        """
        Close the Nacos configuration client and cleanup resources.

        This method should be called during application shutdown to properly
        release connections and stop the listener task.
        """
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self.config_client:
            try:
                await self.config_client.shutdown()
                logger.info("Nacos config client closed")
            except Exception as e:
                logger.warning(f"Error closing Nacos client: {e}")

    @staticmethod
    def from_env() -> "NacosConfigLoader":
        """
        Create a NacosConfigLoader from environment variables.

        Reads Nacos configuration from environment variables with sensible defaults
        for local development.

        Environment variables:
            - NACOS_ENABLED: Enable Nacos configuration (default: true)
            - NACOS_SERVER_ADDRESS: Nacos server address (default: 127.0.0.1:8848)
            - NACOS_NAMESPACE: Nacos namespace (default: agent-breaker-local)
            - NACOS_DATA_ID: Configuration Data ID (default: agent-runner.yaml)
            - NACOS_GROUP: Configuration group (default: DEFAULT_GROUP)
            - NACOS_USERNAME: Nacos username (default: nacos)
            - NACOS_PASSWORD: Nacos password (default: nacos)

        Returns:
            NacosConfigLoader: A new NacosConfigLoader instance configured from environment.
        """
        return NacosConfigLoader(
            enabled=os.getenv("NACOS_ENABLED", "true").lower() == "true",
            server_address=os.getenv("NACOS_SERVER_ADDRESS", "127.0.0.1:8848"),
            namespace=os.getenv("NACOS_NAMESPACE", "agent-breaker-local"),
            data_id=os.getenv("NACOS_DATA_ID", "agent-runner.yaml"),
            group=os.getenv("NACOS_GROUP", "DEFAULT_GROUP"),
            username=os.getenv("NACOS_USERNAME", "nacos"),
            password=os.getenv("NACOS_PASSWORD", "nacos"),
        )


# Global Nacos config loader instance
_nacos_loader: NacosConfigLoader | None = None


async def get_nacos_loader() -> NacosConfigLoader:
    """
    Get or create the global Nacos configuration loader instance.

    Returns:
        NacosConfigLoader: The global Nacos configuration loader.
    """
    global _nacos_loader
    if _nacos_loader is None:
        _nacos_loader = NacosConfigLoader.from_env()
        await _nacos_loader.initialize()
    return _nacos_loader


async def get_nacos_config() -> dict[str, Any]:
    """
    Get configuration from Nacos.

    Returns:
        dict[str, Any]: The configuration dictionary from Nacos.
    """
    loader = await get_nacos_loader()
    return await loader.get_config()


async def close_nacos_loader() -> None:
    """
    Close the global Nacos configuration loader.

    This should be called during application shutdown.
    """
    global _nacos_loader
    if _nacos_loader:
        await _nacos_loader.close()
        _nacos_loader = None