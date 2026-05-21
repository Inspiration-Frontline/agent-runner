from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=CONFIG_DIR / "agent-runner.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "agent-runner"
    debug: bool = Field(default=False, validation_alias="AGENT_RUNNER_DEBUG")

    lite_llm_base_url: str = "http://localhost:4000"
    lite_llm_api_key: str = "sk-agent-breaker-local"

    agent_config_center_url: str = "http://localhost:8081"
    conversation_service_url: str = "http://localhost:8082"
    user_profiler_url: str = "http://localhost:8083"
    knowledge_service_url: str = "http://localhost:8084"
    local_agent_config_enabled: bool = True
    local_agent_config_path: str = str(CONFIG_DIR / "agents.json")

    max_context_tokens: int = 128000
    max_output_tokens: int = 4096


settings = Settings()


class MemoryPolicy(BaseModel):
    profile: bool = True
    rag: bool = True


class ChatRequest(BaseModel):
    agent_id: str
    version: str | None = None
    conversation_id: str | None = None
    user_id: str
    message: str


class AgentConfig(BaseModel):
    agent_id: str
    version: str
    model: str
    system_prompt: str
    tools: list[str] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list)
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)
    max_output_tokens: int = 4096
    temperature: float = 0.7
