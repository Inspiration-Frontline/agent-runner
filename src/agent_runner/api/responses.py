from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health payload returned by the Runner liveness endpoint."""

    status: str
    """Human-readable health state, normally ``healthy``."""


class DebugConfigResponse(BaseModel):
    """Safe configuration projection exposed by development-only diagnostics."""

    lite_llm_base_url: str
    """LiteLLM gateway URL without API key material."""
    agent_config_center_url: str
    """Agent Configuration Center base URL."""
    conversation_service_url: str
    """Conversation Manager HTTP base URL."""
    redis_host: str
    """Redis host used for locks and optional Agent cache."""
    redis_port: int
    """Redis TCP port used for locks and optional Agent cache."""
    agent_config_cache_ttl_seconds: int
    """Configured lifetime of cached Agent definitions."""
    nacos_enabled: bool
    """Whether Nacos configuration refresh is enabled."""
    nacos_namespace: str
    """Nacos namespace supplying effective runtime configuration."""
    environment: str
    """Selected deployment environment name."""
    debug_endpoints_enabled: bool
    """Whether development diagnostic routes are enabled."""
