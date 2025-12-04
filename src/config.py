"""Application configuration using Pydantic settings."""

from typing import Optional

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureOpenAIConfig(BaseSettings):
    """Azure OpenAI configuration with validation."""

    api_key: str
    endpoint: str
    deployment_name: str = "gpt-4o-mini"
    api_version: str = "2024-02-15-preview"
    max_retries: int = 3
    timeout: int = 60

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    @field_validator("api_key", "endpoint")
    @classmethod
    def must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("Value must not be empty")
        return value

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("Azure OpenAI endpoint must use HTTPS")
        return value.rstrip("/")


class VectorDBConfig(BaseSettings):
    """Chroma vector database configuration."""

    host: str
    port: int = 8000
    collection_name: str
    similarity_threshold: float = 0.75
    max_results: int = 5
    connection_timeout: int = 10

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


class ElasticsearchConfig(BaseSettings):
    """Elasticsearch configuration for hybrid deployment."""

    hosts: list[str]
    index_name: str
    username: Optional[str] = None
    password: Optional[str] = None
    use_ssl: bool = True
    verify_certs: bool = True
    timeout: int = 30

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


class MonitoringConfig(BaseSettings):
    """Application Insights and observability configuration."""

    instrumentation_key: str
    log_level: str = "INFO"
    enable_distributed_tracing: bool = True
    sample_rate: float = 1.0

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


class WorkflowConfig(BaseSettings):
    """Workflow execution parameters."""

    max_retries: int = 3
    retry_delay: int = 5
    execution_timeout: int = 300
    enable_checkpointing: bool = True
    checkpoint_backend: str = "redis"
    redis_url: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


# Global configuration instances
azure_config = AzureOpenAIConfig()
vector_db_config = VectorDBConfig()
es_config = ElasticsearchConfig()
monitoring_config = MonitoringConfig()
workflow_config = WorkflowConfig()
