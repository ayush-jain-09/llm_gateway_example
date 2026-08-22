"""
Round-Robin Routing Strategy.
Cycles through available providers evenly in circular sequence.
Time Complexity: O(1)
"""

import asyncio
from typing import List
from src.router.base import RouterBase
from src.providers.base import BaseProvider
from src.gateway.models import ChatRequest


class RoundRobinRouter(RouterBase):
    def __init__(self):
        self._index: int = 0
        self._lock = asyncio.Lock()

    async def select_provider(
        self,
        available_providers: List[BaseProvider],
        request: ChatRequest,
    ) -> BaseProvider:
        if not available_providers:
            raise RuntimeError("No healthy providers available for routing")

        # Explicit preference override
        if request.preferred_provider:
            for p in available_providers:
                if p.provider_id == request.preferred_provider:
                    return p

        async with self._lock:
            selected = available_providers[self._index % len(available_providers)]
            self._index = (self._index + 1) % len(available_providers)
            return selected
