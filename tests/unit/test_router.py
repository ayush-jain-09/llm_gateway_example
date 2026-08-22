"""
Unit tests for Router and Load Balancing strategies.
Tests: Round Robin, EWMA Latency-Aware routing, Cost-Aware optimization, and Consistent Hashing Ring.
"""

import pytest
from src.router.round_robin import RoundRobinRouter
from src.router.latency_aware import LatencyAwareEWMARouter
from src.router.cost_aware import CostAwareRouter
from src.router.consistent_hash import ConsistentHashRouter
from src.providers.mock_adapter import MockProvider
from src.gateway.config import ProviderConfig
from src.gateway.models import ChatRequest, ChatMessage


def _make_providers():
    p1 = MockProvider(ProviderConfig(
        id="provider-cheap-slow",
        type="mock",
        model="m1",
        cost_per_1k_input_tokens=0.0001,
        cost_per_1k_output_tokens=0.0001,
        simulated_latency_ms=300,
    ))
    p2 = MockProvider(ProviderConfig(
        id="provider-expensive-fast",
        type="mock",
        model="m2",
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.005,
        simulated_latency_ms=50,
    ))
    return [p1, p2]


@pytest.mark.asyncio
async def test_round_robin_cycling():
    providers = _make_providers()
    router = RoundRobinRouter()
    req = ChatRequest(messages=[ChatMessage(content="hi")])

    sel1 = await router.select_provider(providers, req)
    sel2 = await router.select_provider(providers, req)
    sel3 = await router.select_provider(providers, req)

    assert sel1.provider_id == "provider-cheap-slow"
    assert sel2.provider_id == "provider-expensive-fast"
    assert sel3.provider_id == "provider-cheap-slow"


@pytest.mark.asyncio
async def test_cost_aware_routing():
    providers = _make_providers()
    router = CostAwareRouter()
    req = ChatRequest(messages=[ChatMessage(content="hi")])

    selected = await router.select_provider(providers, req)
    assert selected.provider_id == "provider-cheap-slow"


@pytest.mark.asyncio
async def test_latency_aware_routing():
    providers = _make_providers()
    router = LatencyAwareEWMARouter(alpha=0.5)
    req = ChatRequest(messages=[ChatMessage(content="hi")])

    # Record provider 1 as slow (500ms) and provider 2 as fast (20ms)
    await router.record_feedback("provider-cheap-slow", 500.0, success=True)
    await router.record_feedback("provider-expensive-fast", 20.0, success=True)

    selected = await router.select_provider(providers, req)
    assert selected.provider_id == "provider-expensive-fast"


@pytest.mark.asyncio
async def test_consistent_hash_stickiness():
    providers = _make_providers()
    router = ConsistentHashRouter(virtual_nodes_per_provider=20)
    
    req_tenant1 = ChatRequest(messages=[ChatMessage(content="query 1")], tenant_id="tenant_alpha")
    req_tenant2 = ChatRequest(messages=[ChatMessage(content="query 2")], tenant_id="tenant_alpha")

    # Same tenant must route to the same provider
    sel1 = await router.select_provider(providers, req_tenant1)
    sel2 = await router.select_provider(providers, req_tenant2)
    assert sel1.provider_id == sel2.provider_id
