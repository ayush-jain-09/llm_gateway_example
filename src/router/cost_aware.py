"""
Cost-Aware Routing Strategy.
Optimizes financial spend by prioritizing the cheapest available provider for a given model or general task.
Time Complexity: O(K) where K = number of available providers.
"""

from typing import List
from src.router.base import RouterBase
from src.providers.base import BaseProvider
from src.gateway.models import ChatRequest


class CostAwareRouter(RouterBase):
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

        # Sort by total blended token cost (input + output)
        cheapest_provider = min(
            available_providers,
            key=lambda p: (p.cost_per_1k_input + p.cost_per_1k_output),
        )
        return cheapest_provider
