"""Configuration management for DevOps Agent."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class ServerConfig(BaseSettings):
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    url: str = "sqlite:///./devops_agent.db"
    pool_size: int = 10
    pool_pre_ping: bool = True
    echo: bool = False


class LLMConfig(BaseSettings):
    """LLM provider configuration."""
    provider: str = "openai"  # openai or anthropic
    temperature: float = 0.1
    max_tokens: int = 4096
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


class MCPServerConfig(BaseSettings):
    """Individual MCP server configuration."""
    server_id: str
    transport: str = "http"
    endpoint: str
    token: Optional[str] = None
    timeout_seconds: int = 30


class AgentConfig(BaseSettings):
    """Individual agent configuration."""
    timeout_seconds: int = 30
    max_retries: int = 3


class SecurityConfig(BaseSettings):
    """Security configuration."""
    cors_enabled: bool = True
    cors_origins: list[str] = ["http://localhost:3000"]
    auth_enabled: bool = False
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    token_expire_minutes: int = 60


class LimitsConfig(BaseSettings):
    """System limits configuration."""
    max_concurrent_agents: int = 3
    max_evidence_per_request: int = 100
    max_tasks_per_plan: int = 20
    default_timeout_seconds: int = 30
    max_retries: int = 3
    max_request_size_mb: int = 10


class ObservabilityConfig(BaseSettings):
    """Observability configuration."""
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "logs/devops-agent.log"
    tracing_enabled: bool = True
    metrics_enabled: bool = True
    metrics_port: int = 9090


class Settings(BaseSettings):
    """Main application settings."""
    
    # Load from environment and config file
    environment: str = Field(default="development", alias="API_ENV")
    config_file: str = Field(default="config/default.yaml")
    
    # Sub-configurations
    server: ServerConfig = ServerConfig()
    database: DatabaseConfig = DatabaseConfig()
    llm: LLMConfig = LLMConfig()
    security: SecurityConfig = SecurityConfig()
    limits: LimitsConfig = LimitsConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    
    # Agent configurations
    agents: Dict[str, AgentConfig] = {
        "planner": AgentConfig(timeout_seconds=30, max_retries=2),
        "log": AgentConfig(timeout_seconds=20, max_retries=3),
        "alarm": AgentConfig(timeout_seconds=15, max_retries=3),
        "kpi": AgentConfig(timeout_seconds=20, max_retries=3),
        "summary": AgentConfig(timeout_seconds=40, max_retries=2),
    }
    
    # MCP configurations
    mcp_servers: Dict[str, MCPServerConfig] = {}
    mcp_mapping: Dict[str, Dict[str, Any]] = {}
    
    # LangGraph settings
    langgraph_max_steps: int = 20
    langgraph_memory_type: str = "sqlite"
    langgraph_checkpoint_ttl_seconds: int = 3600
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    cache_max_size_mb: int = 100
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_size: int = 10
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_yaml_config()
        self._setup_mcp_servers()
    
    def _load_yaml_config(self):
        """Load configuration from YAML file."""
        config_path = PROJECT_ROOT / self.config_file
        if not config_path.exists():
            return
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        # Override with YAML values if present
        if "agents" in config:
            for agent_name, agent_config in config["agents"].items():
                self.agents[agent_name] = AgentConfig(**agent_config)
                
        if "mcp" in config and "mapping" in config["mcp"]:
            self.mcp_mapping = config["mcp"]["mapping"]
    
    def _setup_mcp_servers(self):
        """Setup MCP servers from environment variables."""
        # Logs server
        if os.getenv("MCP_LOGS_SERVER_URL"):
            self.mcp_servers["logs"] = MCPServerConfig(
                server_id="logs",
                endpoint=os.getenv("MCP_LOGS_SERVER_URL"),
                token=os.getenv("MCP_LOGS_TOKEN"),
            )
        
        # Metrics server
        if os.getenv("MCP_METRICS_SERVER_URL"):
            self.mcp_servers["metrics"] = MCPServerConfig(
                server_id="metrics",
                endpoint=os.getenv("MCP_METRICS_SERVER_URL"),
                token=os.getenv("MCP_METRICS_TOKEN"),
            )
        
        # Alerts server
        if os.getenv("MCP_ALERTS_SERVER_URL"):
            self.mcp_servers["alerts"] = MCPServerConfig(
                server_id="alerts",
                endpoint=os.getenv("MCP_ALERTS_SERVER_URL"),
                token=os.getenv("MCP_ALERTS_TOKEN"),
            )
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for the selected provider."""
        if self.llm.provider == "openai":
            return {
                "api_key": self.llm.openai_api_key,
                "model": self.llm.openai_model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            }
        elif self.llm.provider == "anthropic":
            return {
                "api_key": self.llm.anthropic_api_key,
                "model": self.llm.anthropic_model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm.provider}")
    
    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_name, AgentConfig())
    
    def get_mcp_server_config(self, server_id: str) -> Optional[MCPServerConfig]:
        """Get configuration for a specific MCP server."""
        return self.mcp_servers.get(server_id)
    
    @property
    def database_url(self) -> str:
        """Get the database URL."""
        return self.database.url
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


# Global settings instance
settings = Settings()


# Export commonly used configurations
def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def get_llm_client():
    """Get configured LLM client based on settings."""
    config = settings.get_llm_config()
    
    if settings.llm.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=config["api_key"],
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )
    elif settings.llm.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=config["api_key"],
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")