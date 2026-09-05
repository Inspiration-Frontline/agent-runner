"""
Nacos configuration loader module.

This module provides functionality to load configuration from Nacos configuration center,
with support for YAML format, configuration caching, and dynamic refresh through listeners.
"""

import asyncio
import contextlib
import logging
import os
from typing import Any

import yaml
from v2.nacos import ClientConfigBuilder, ConfigParam, GRPCConfig, NacosConfigService

from agent_runner.config import Settings

logger = logging.getLogger(__name__)


class NacosConfigLoader:
    """Nacos configuration loader with async support.

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
        server_address: Nacos server address used to create the configuration client.
        username: Optional Nacos account used for authenticated configuration reads.
        password: Optional Nacos password paired with ``username``.
        _cached_config: Validated cached configuration.
        _configuration_revision: Monotonic configuration snapshot revision.
        _listener_task: Lifecycle-owned asynchronous task for listener.
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
    ) -> None:
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
        # Key: Nacos configuration property name. Value: latest parsed remote property value.
        self._cached_config: dict[str, Any] = {}
        self._configuration_revision = 0
        self._listener_task: asyncio.Task[None] | None = None

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
            client_config: Any = (
                ClientConfigBuilder()
                .server_address(self.server_address)
                .namespace_id(self.namespace)
                .username(self.username)
                .password(self.password)
                # The SDK logs access tokens and owns a process-global rolling file handler.
                # Application logs provide the credential-safe lifecycle and failure evidence.
                .log_level("CRITICAL")
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
            config_param: Any = ConfigParam(data_id=self.data_id, group=self.group)
            content: Any = await self.config_client.get_config(config_param)

            if content:
                parsed_config: dict[str, Any] = self._parse_and_replace_config(content)
                logger.debug(f"Loaded configuration from Nacos: {self.data_id}")

                return parsed_config

        except Exception as error:
            logger.warning(
                "Failed to load configuration from Nacos",
                extra={"error_type": type(error).__name__},
            )

        return {}

    async def _start_config_listener(self) -> None:
        """
        Start listening for configuration changes in Nacos.

        This method runs as a background task and updates the cached configuration
        whenever changes are detected in Nacos.
        """

        if not self.config_client:
            return

        try:
            await self.config_client.add_listener(
                data_id=self.data_id,
                group=self.group,
                listener=self._handle_config_change,
            )
            logger.info(f"Started configuration listener for {self.data_id}")
        except Exception as e:
            logger.warning(f"Failed to add configuration listener: {e}")

    async def _handle_config_change(self, namespace_id: str, group: str, data_id: str, content: str) -> None:
        """Apply one Nacos change callback after parsing and validating the complete document.

        Args:
            namespace_id: Nacos namespace that emitted the callback.
            group: Nacos configuration group that emitted the callback.
            data_id: Nacos Data ID that changed.
            content: Replacement YAML document, possibly empty when the Data ID is removed.
        """
        logger.info(f"Configuration changed in Nacos: data_id={data_id}, group={group}")

        try:
            # An empty callback means the Data ID was cleared or removed. Publishing an empty
            # snapshot revokes cached Secrets instead of retaining credentials that no longer
            # exist in the configuration center.
            self._parse_and_replace_config(content)
            logger.info("Configuration cache updated from Nacos")
        except Exception as error:
            logger.warning(
                "Failed to parse updated Nacos configuration",
                extra={"error_type": type(error).__name__},
            )

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

    @property
    def cached_config(self) -> dict[str, Any]:
        """Return the latest parsed Nacos document used for synchronous settings reads.

        Returns:
            The latest parsed Nacos document used for synchronous settings reads.
        """

        return self._cached_config

    @property
    def configuration_revision(self) -> int:
        """Return the monotonic revision assigned to the latest valid Nacos snapshot.

        Returns:
            The monotonic revision assigned to the latest valid Nacos snapshot.
        """

        return self._configuration_revision

    @staticmethod
    def _parse_config(content: str) -> dict[str, Any]:
        """Parse one YAML document and reject non-object roots before publishing it.

        Args:
            content: Text or serialized content processed by the operation.

        Returns:
            Parsed one YAML document and reject non-object roots before publishing it.
        """
        parsed_config: Any | dict[Any, Any] = yaml.safe_load(content) or {}
        if not isinstance(parsed_config, dict):
            raise ValueError("Nacos configuration root must be a YAML object")

        return parsed_config

    def _replace_cached_config(self, parsed_config: dict[str, Any]) -> None:
        """Atomically publish one valid snapshot and increment its monotonic revision.

        Args:
            parsed_config: Validated configuration snapshot ready for atomic publication.
        """
        self._cached_config = parsed_config
        self._configuration_revision += 1

    def _parse_and_replace_config(self, content: str) -> dict[str, Any]:
        """Parse a complete Nacos document before atomically publishing its new revision.

        Args:
            content: Text or serialized content processed by the operation.

        Returns:
            Parsed a complete Nacos document before atomically publishing its new revision.
        """
        parsed_config: dict[str, Any] = self._parse_config(content)
        self._replace_cached_config(parsed_config)

        return parsed_config

    async def close(self) -> None:
        """
        Close the Nacos configuration client and cleanup resources.

        This method should be called during application shutdown to properly
        release connections and stop the listener task.
        """

        if self._listener_task:
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task

        if self.config_client:
            try:
                await self.config_client.shutdown()
                logger.info("Nacos config client closed")
            except Exception as e:
                logger.warning(f"Error closing Nacos client: {e}")

    @staticmethod
    def from_settings(settings: Settings) -> "NacosConfigLoader":
        """
        Create a NacosConfigLoader from application settings.

        The Settings instance already merges OS environment values with the
        repo-local env file, so this keeps PyCharm/direct-file startup and
        verification runs on the same configuration path.

        Args:
            settings: Application settings loaded from env files and environment.

        Returns:
            NacosConfigLoader: A new loader configured from application settings.
        """

        return NacosConfigLoader(
            enabled=settings.nacos_enabled,
            server_address=settings.nacos_server_address,
            namespace=settings.nacos_namespace,
            data_id=settings.nacos_data_id,
            group=settings.nacos_group,
            username=settings.nacos_username,
            password=settings.nacos_password,
        )

    @staticmethod
    def from_env() -> "NacosConfigLoader":
        """
        Create a NacosConfigLoader from environment variables.

        Reads Nacos configuration from environment variables with sensible defaults

        for local development.

        Environment variables:
            - NACOS_ENABLED: Enable Nacos configuration (default: false)
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
            enabled=os.getenv("NACOS_ENABLED", "false").lower() == "true",
            server_address=os.getenv("NACOS_SERVER_ADDRESS", "127.0.0.1:8848"),
            namespace=os.getenv("NACOS_NAMESPACE", "agent-breaker-local"),
            data_id=os.getenv("NACOS_DATA_ID", "agent-runner.yaml"),
            group=os.getenv("NACOS_GROUP", "DEFAULT_GROUP"),
            username=os.getenv("NACOS_USERNAME", "nacos"),
            password=os.getenv("NACOS_PASSWORD", "nacos"),
        )
