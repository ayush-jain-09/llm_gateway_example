"""
Factory for Router / Load Balancer Strategy instantiation.
Implements Creational Factory Pattern.
"""

from src.router.base import RouterBase
from src.router.round_robin import RoundRobinRouter
from src.router.latency_aware import LatencyAwareEWMARouter
from src.router.cost_aware import CostAwareRouter
from src.router.consistent_hash import ConsistentHashRouter
from src.gateway.config import RouterConfig


class RouterFactory:
    """Creates router instances based on configuration."""

    @staticmethod
    def create(config: RouterConfig) -> RouterBase:
        strategy = config.strategy.lower()

        if strategy == "round_robin":
            return RoundRobinRouter()
        elif strategy in ("latency_aware", "latency_aware_ewma"):
            return LatencyAwareEWMARouter(alpha=config.ewma_alpha)
        elif strategy == "cost_aware":
            return CostAwareRouter()
        elif strategy == "consistent_hashing":
            return ConsistentHashRouter(
                virtual_nodes_per_provider=config.virtual_nodes_per_provider
            )
        else:
            raise ValueError(f"Unsupported routing strategy: '{strategy}'")
