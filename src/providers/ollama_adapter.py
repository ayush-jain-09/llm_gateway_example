"""
Ollama Local Model Provider Adapter.
Implements the Adapter Pattern for local Ollama instance (e.g. Llama 3, Mistral).
"""

import time
import httpx
from src.providers.base import BaseProvider
from src.gateway.models import ChatRequest, ChatResponse, ChatChoice, ChatMessage, UsageInfo
from src.gateway.config import ProviderConfig


class OllamaAdapter(BaseProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = (config.base_url or "http://localhost:11434").rstrip("/")

    async def generate_completion(self, request: ChatRequest) -> ChatResponse:
        start_time = time.monotonic()
        url = f"{self.base_url}/api/chat"

        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
        ]

        payload = {
            "model": request.model or self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama API error ({resp.status_code}): {resp.text}"
                )
            data = resp.json()

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        msg_data = data.get("message", {})
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return ChatResponse(
            model=data.get("model", self.model_name),
            provider_id=self.provider_id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role=msg_data.get("role", "assistant"),
                        content=msg_data.get("content", ""),
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=0.0,
            ),
            latency_ms=round(elapsed_ms, 2),
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
