"""
Pydantic configuration loader for LLM Gateway.
Supports hot-reloading and environment variable overrides.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
import yaml
from pathlib import Path
import os


class GatewayServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    enable_prometheus: bool = True


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "llm_gw:rate_limit:"


class RateLimiterConfig(BaseModel):
    strategy: Literal[
        "token_bucket",
        "sliding_window_log",
        "sliding_window_counter",
        "redis_token_bucket",
    ] = "token_bucket"
    default_capacity: int = 10
    default_refill_rate: float = 2.0
    sliding_window_seconds: int = 60
    redis: RedisConfig = Field(default_factory=RedisConfig)


class CacheConfig(BaseModel):
    strategy: Literal[
        "disabled",
        "lru_exact",
        "lfu_exact",
        "semantic_vector",
    ] = "semantic_vector"
    max_size: int = 1000
    ttl_seconds: int = 3600
    similarity_threshold: float = 0.88
    embedding_model: str = "all-MiniLM-L6-v2"


class RouterConfig(BaseModel):
    strategy: Literal[
        "round_robin",
        "latency_aware_ewma",
        "cost_aware",
        "consistent_hashing",
    ] = "latency_aware_ewma"
    ewma_alpha: float = 0.2
    virtual_nodes_per_provider: int = 50


class CircuitBreakerConfig(BaseModel):
    strategy: Literal[
        "consecutive_failure",
        "sliding_window_error_rate",
    ] = "sliding_window_error_rate"
    failure_rate_threshold: float = 0.5
    window_size_seconds: float = 30.0
    minimum_requests: int = 5
    recovery_time_seconds: float = 15.0
    half_open_success_threshold: int = 3


class ProviderConfig(BaseModel):
    id: str
    type: Literal["openai", "gemini", "perplexity", "ollama", "mock"]
    enabled: bool = True
    model: str
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    simulated_latency_ms: int = 100
    simulated_failure_rate: float = 0.0
    timeout_seconds: float = 10.0


class AppConfig(BaseModel):
    gateway: GatewayServerConfig = Field(default_factory=GatewayServerConfig)
    rate_limiter: RateLimiterConfig = Field(default_factory=RateLimiterConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    providers: List[ProviderConfig] = Field(default_factory=list)
    failover_order: List[str] = Field(default_factory=list)

    @classmethod
    def load_from_yaml(cls, path: str = "config/gateway_config.yaml") -> "AppConfig":
        config_path = Path(path)
        if not config_path.is_absolute():
            # Find relative to workspace root
            root = Path(__file__).resolve().parent.parent.parent
            config_path = root / path
            
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        return cls(**raw_data)
