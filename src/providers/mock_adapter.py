"""
Mock LLM Provider Adapter.
Used for zero-cost offline testing, load testing, and failure injection without API keys.
"""

import time
import asyncio
import random
from typing import Optional
from src.providers.base import BaseProvider
from src.gateway.models import ChatRequest, ChatResponse, ChatChoice, ChatMessage, UsageInfo
from src.gateway.config import ProviderConfig


class MockProvider(BaseProvider):
    async def generate_completion(self, request: ChatRequest) -> ChatResponse:
        start_time = time.monotonic()

        # Simulate latency
        if self.config.simulated_latency_ms > 0:
            await asyncio.sleep(self.config.simulated_latency_ms / 1000.0)

        # Simulate failure rate for circuit breaker testing
        if self.config.simulated_failure_rate > 0.0:
            if random.random() < self.config.simulated_failure_rate:
                raise ConnectionError(
                    f"Simulated fault injection failure from provider '{self.provider_id}'"
                )

        last_user_message = "Hello"
        for msg in reversed(request.messages):
            if msg.role == "user":
                last_user_message = msg.content
                break

        response_content = (
            f"[{self.provider_id} / {self.model_name}] Response to: '{last_user_message[:50]}'"
        )

        prompt_tokens = sum(len(m.content.split()) for m in request.messages) + 10
        completion_tokens = len(response_content.split())
        cost = self.calculate_cost(prompt_tokens, completion_tokens)
        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        return ChatResponse(
            model=self.model_name,
            provider_id=self.provider_id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=cost,
            ),
            latency_ms=round(elapsed_ms, 2),
        )

    async def health_check(self) -> bool:
        return True
