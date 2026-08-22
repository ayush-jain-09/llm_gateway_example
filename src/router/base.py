"""
Abstract Base Class for Routing and Load Balancing Strategies.
Implements the Strategy Pattern (Behavioral).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from src.providers.base import BaseProvider
from src.gateway.models import ChatRequest


class RouterBase(ABC):
    """
    Abstract interface for provider selection and traffic distribution strategies.
    """

    @abstractmethod
    async def select_provider(
        self,
        available_providers: List[BaseProvider],
        request: ChatRequest,
    ) -> BaseProvider:
        """
        Selects the optimal provider candidate from a list of currently healthy providers.
        """
        pass

    async def record_feedback(
        self,
        provider_id: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Optional hook for adaptive algorithms to update internal stats."""
        pass
