"""
Google Gemini Provider Adapter using Google REST API.
Translates standard OpenAI ChatRequest format into Gemini's generateContent JSON schema.
"""

import os
import time
import httpx
from src.providers.base import BaseProvider
from src.gateway.models import ChatRequest, ChatResponse, ChatChoice, ChatMessage, UsageInfo
from src.gateway.config import ProviderConfig


class GeminiAdapter(BaseProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        env_var = config.api_key_env or "GEMINI_API_KEY"
        self.api_key = os.getenv(env_var, "")

    def _convert_messages_to_gemini(self, messages):
        """Converts OpenAI role/content schema to Gemini contents/parts schema."""
        contents = []
        for msg in messages:
            role = "user" if msg.role in ("user", "system") else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })
        return contents

    async def generate_completion(self, request: ChatRequest) -> ChatResponse:
        start_time = time.monotonic()
        model_id = request.model or self.model_name
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={self.api_key}"

        payload = {
            "contents": self._convert_messages_to_gemini(request.messages),
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            }
        }

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Gemini API error ({resp.status_code}): {resp.text}"
                )
            data = resp.json()

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini API returned empty candidate response")

        text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        usage_meta = data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        completion_tokens = usage_meta.get("candidatesTokenCount", 0)
        cost = self.calculate_cost(prompt_tokens, completion_tokens)

        return ChatResponse(
            model=model_id,
            provider_id=self.provider_id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text_content),
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
        return bool(self.api_key)
