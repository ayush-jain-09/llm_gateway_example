"""
Abstract Base Class for LLM Provider Adapters.
Implements the Adapter Pattern (Structural).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time
from src.gateway.models import ChatRequest, ChatResponse, ChatChoice, ChatMessage, UsageInfo
from src.gateway.config import ProviderConfig


class BaseProvider(ABC):
    """
    Uniform adapter interface for all external and local LLM services.
    Converts gateway ChatRequest into provider-specific REST format and standardizes responses.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.provider_id = config.id
        self.model_name = config.model
        self.cost_per_1k_input = config.cost_per_1k_input_tokens
        self.cost_per_1k_output = config.cost_per_1k_output_tokens

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = (prompt_tokens / 1000.0) * self.cost_per_1k_input
        output_cost = (completion_tokens / 1000.0) * self.cost_per_1k_output
        return round(input_cost + output_cost, 6)

    @abstractmethod
    async def generate_completion(self, request: ChatRequest) -> ChatResponse:
        """
        Executes chat completion against upstream LLM service.
        Must raise standard exceptions (TimeoutError, ConnectionError, ValueError) on failures.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Quick connectivity probe."""
        pass
