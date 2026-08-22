"""
Latency-Aware Load Balancing Strategy using Exponentially Weighted Moving Average (EWMA).

Algorithm:
- Tracks EWMA latency per provider: EWMA_t = alpha * latency + (1 - alpha) * EWMA_(t-1)
- Routes incoming request to the provider with the lowest current EWMA latency.
- Dynamic adaptation to provider degradation or slowdowns.
- Time Complexity: O(K) where K = number of available providers.
"""

import asyncio
from typing import List, Dict
from src.router.base import RouterBase
from src.providers.base import BaseProvider
from src.gateway.models import ChatRequest


class LatencyAwareEWMARouter(RouterBase):
    def __init__(self, alpha: float = 0.2, default_initial_latency_ms: float = 200.0):
        self.alpha = max(0.01, min(1.0, alpha))
        self.default_initial_latency_ms = default_initial_latency_ms
        self._ewma_latencies: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def record_feedback(
        self,
        provider_id: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        # Penalize errors by treating as 5000ms latency spike
        effective_latency = latency_ms if success else max(5000.0, latency_ms * 2.0)
        async with self._lock:
            if provider_id not in self._ewma_latencies:
                self._ewma_latencies[provider_id] = effective_latency
            else:
                prev = self._ewma_latencies[provider_id]
                self._ewma_latencies[provider_id] = (self.alpha * effective_latency) + (
                    (1.0 - self.alpha) * prev
                )

    async def select_provider(
        self,
        available_providers: List[BaseProvider],
        request: ChatRequest,
    ) -> BaseProvider:
        if not available_providers:
            raise RuntimeError("No healthy providers available for routing")

        if request.preferred_provider:
            for p in available_providers:
                if p.provider_id == request.preferred_provider:
                    return p

        async with self._lock:
            best_provider = available_providers[0]
            best_latency = self._ewma_latencies.get(
                best_provider.provider_id, self.default_initial_latency_ms
            )

            for provider in available_providers[1:]:
                lat = self._ewma_latencies.get(
                    provider.provider_id, self.default_initial_latency_ms
                )
                if lat < best_latency:
                    best_latency = lat
                    best_provider = provider

            return best_provider
