"""
Consistent Hashing Ring with Virtual Nodes.

Algorithm:
- Maps providers to multiple virtual points along a 32-bit integer ring [0, 2^32 - 1].
- Keys (e.g. tenant_id or session_id) are hashed onto the ring.
- Binary search (`bisect_right`) locates the nearest clockwise node in O(log(V * N)) time.
- Guarantees sticky routing for conversation cache reuse and smooth rebalancing on node changes.
"""

import hashlib
import bisect
from typing import List, Dict, Tuple, Optional
from src.router.base import RouterBase
from src.providers.base import BaseProvider
from src.gateway.models import ChatRequest


class ConsistentHashRouter(RouterBase):
    def __init__(self, virtual_nodes_per_provider: int = 50):
        self.virtual_nodes_per_provider = max(1, virtual_nodes_per_provider)
        self._ring: List[int] = []
        self._ring_map: Dict[int, str] = {}  # hash -> provider_id

    def _hash(self, key: str) -> int:
        """32-bit MD5 integer hash."""
        return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)

    def _build_ring(self, available_providers: List[BaseProvider]) -> None:
        self._ring.clear()
        self._ring_map.clear()

        for provider in available_providers:
            for v_idx in range(self.virtual_nodes_per_provider):
                v_key = f"{provider.provider_id}#vnode_{v_idx}"
                v_hash = self._hash(v_key)
                self._ring.append(v_hash)
                self._ring_map[v_hash] = provider.provider_id

        self._ring.sort()

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

        # Rebuild ring if provider set changed
        self._build_ring(available_providers)

        # Routing key: session_id > tenant_id > default
        routing_key = (
            request.session_id
            or request.tenant_id
            or (request.messages[-1].content if request.messages else "default")
        )

        key_hash = self._hash(routing_key)

        # Binary search for the first node with hash >= key_hash
        idx = bisect.bisect_right(self._ring, key_hash)
        if idx == len(self._ring):
            idx = 0  # Wrap around ring

        selected_hash = self._ring[idx]
        selected_provider_id = self._ring_map[selected_hash]

        for p in available_providers:
            if p.provider_id == selected_provider_id:
                return p

        return available_providers[0]
