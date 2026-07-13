from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class DebugConfigResponse(BaseModel):
    lite_llm_base_url: str
    agent_config_center_url: str
    conversation_service_url: str
    redis_host: str
    redis_port: int
    agent_config_cache_ttl_seconds: int
    nacos_enabled: bool
    nacos_namespace: str
    environment: str
    debug_endpoints_enabled: bool
