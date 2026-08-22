"""
OpenAI Provider Adapter using async HTTP client.
Implements the Adapter Pattern for OpenAI's /v1/chat/completions endpoint.
"""

import os
import time
import httpx
from src.providers.base import BaseProvider
from src.gateway.models import ChatRequest, ChatResponse, ChatChoice, ChatMessage, UsageInfo
from src.gateway.config import ProviderConfig


class OpenAIAdapter(BaseProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        env_var = config.api_key_env or "OPENAI_API_KEY"
        self.api_key = os.getenv(env_var, "")
        self.base_url = (config.base_url or "https://api.openai.com/v1").rstrip("/")

    async def generate_completion(self, request: ChatRequest) -> ChatResponse:
        start_time = time.monotonic()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": request.model or self.model_name,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenAI API error ({resp.status_code}): {resp.text}"
                )
            data = resp.json()

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        choice_data = data["choices"][0]
        usage_data = data.get("usage", {})
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        cost = self.calculate_cost(prompt_tokens, completion_tokens)

        return ChatResponse(
            id=data.get("id"),
            model=data.get("model", self.model_name),
            provider_id=self.provider_id,
            choices=[
                ChatChoice(
                    index=choice_data.get("index", 0),
                    message=ChatMessage(
                        role=choice_data["message"]["role"],
                        content=choice_data["message"]["content"],
                    ),
                    finish_reason=choice_data.get("finish_reason", "stop"),
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
        return bool(self.api_key)
