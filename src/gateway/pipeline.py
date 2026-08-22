"""
Master Gateway Pipeline Execution Engine.
Orchestrates Rate Limiting -> Caching -> Routing -> Circuit Breaker -> Provider Adapters -> Failovers.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional
from fastapi import HTTPException

from src.gateway.models import ChatRequest, ChatResponse, ChatChoice, ChatMessage, UsageInfo
from src.gateway.config import AppConfig
from src.rate_limiter.base import RateLimiterBase
from src.rate_limiter.factory import RateLimiterFactory
from src.cache.base import CacheBase
from src.cache.factory import CacheFactory
from src.circuit_breaker.breaker import CircuitBreaker
from src.circuit_breaker.states import CircuitState, CircuitBreakerOpenException
from src.router.base import RouterBase
from src.router.factory import RouterFactory
from src.providers.base import BaseProvider
from src.providers.factory import ProviderFactory
from src.observability import metrics

logger = logging.getLogger("llm_gateway.pipeline")


class GatewayPipeline:
    def __init__(self, config: AppConfig):
        self.config = config

        # 1. Initialize Rate Limiter
        self.rate_limiter: RateLimiterBase = RateLimiterFactory.create(config.rate_limiter)

        # 2. Initialize Cache Engine
        self.cache: CacheBase = CacheFactory.create(config.cache)

        # 3. Initialize Router Strategy
        self.router: RouterBase = RouterFactory.create(config.router)

        # 4. Initialize Provider Adapters and per-provider Circuit Breakers
        self.providers: Dict[str, BaseProvider] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        for p_cfg in config.providers:
            if p_cfg.enabled:
                provider = ProviderFactory.create(p_cfg)
                self.providers[p_cfg.id] = provider
                self.circuit_breakers[p_cfg.id] = CircuitBreaker(
                    provider_id=p_cfg.id,
                    config=config.circuit_breaker,
                )

        if not self.providers:
            # Fallback mock provider if none enabled
            from src.gateway.config import ProviderConfig
            fallback_cfg = ProviderConfig(
                id="default-mock",
                type="mock",
                model="mock-model",
                enabled=True,
            )
            self.providers["default-mock"] = ProviderFactory.create(fallback_cfg)
            self.circuit_breakers["default-mock"] = CircuitBreaker("default-mock")

        self.failover_order: List[str] = config.failover_order or list(self.providers.keys())

    def _extract_prompt_text(self, request: ChatRequest) -> str:
        """Flattens conversation messages into a normalized prompt string for hashing and embeddings."""
        return "\n".join(f"{m.role}: {m.content.strip()}" for m in request.messages)

    async def _execute_provider_with_circuit_breaker(
        self, provider: BaseProvider, request: ChatRequest
    ) -> ChatResponse:
        breaker = self.circuit_breakers[provider.provider_id]
        await breaker.allow_request()

        start_time = time.monotonic()
        try:
            response = await provider.generate_completion(request)
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            await breaker.record_success()
            await self.router.record_feedback(provider.provider_id, elapsed_ms, success=True)
            return response
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            await breaker.record_failure(e)
            await self.router.record_feedback(provider.provider_id, elapsed_ms, success=False)
            raise

    async def process_chat_completion(self, request: ChatRequest) -> ChatResponse:
        pipeline_start = time.monotonic()
        tenant = request.tenant_id or "default_tenant"
        prompt_text = self._extract_prompt_text(request)

        # -------------------------------------------------------------
        # STEP 1: Rate Limiting
        # -------------------------------------------------------------
        allowed, remaining, retry_after = await self.rate_limiter.is_allowed(tenant, cost=1)
        if not allowed:
            metrics.RATE_LIMITED_REQUESTS_TOTAL.labels(
                tenant_id=tenant, strategy=self.config.rate_limiter.strategy
            ).inc()
            metrics.GATEWAY_REQUESTS_TOTAL.labels(status="429", tenant_id=tenant).inc()
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for tenant '{tenant}'. Retry after {retry_after:.2f} seconds.",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        # -------------------------------------------------------------
        # STEP 2: Caching (Exact + Semantic Vector Match)
        # -------------------------------------------------------------
        if not request.cache_bypass and self.config.cache.strategy != "disabled":
            cached_item = await self.cache.get(prompt_text)
            if cached_item is not None:
                total_duration = time.monotonic() - pipeline_start
                cache_type = "SEMANTIC" if cached_item.prompt_text != prompt_text else "EXACT"

                metrics.CACHE_HITS_TOTAL.labels(cache_type=cache_type).inc()
                metrics.GATEWAY_REQUESTS_TOTAL.labels(status="200", tenant_id=tenant).inc()
                metrics.GATEWAY_REQUEST_DURATION_SECONDS.labels(
                    provider_id="cache", cache_hit="true"
                ).observe(total_duration)

                # Estimated saved cost
                saved_cost = 0.002
                if hasattr(cached_item.value, "usage"):
                    saved_cost = cached_item.value.usage.cost_usd
                metrics.CACHE_ESTIMATED_DOLLARS_SAVED.inc(saved_cost)

                resp: ChatResponse = cached_item.value
                resp.cache_hit = True
                resp.cache_type = cache_type
                resp.latency_ms = round(total_duration * 1000.0, 2)
                return resp

        metrics.CACHE_MISSES_TOTAL.inc()

        # -------------------------------------------------------------
        # STEP 3: Route Selection & Circuit Breaker Filtering
        # -------------------------------------------------------------
        healthy_providers: List[BaseProvider] = []
        for pid, provider in self.providers.items():
            cb = self.circuit_breakers[pid]
            # Update Prometheus circuit state gauge (0=CLOSED, 1=HALF_OPEN, 2=OPEN)
            state_val = 0 if cb.state == CircuitState.CLOSED else (1 if cb.state == CircuitState.HALF_OPEN else 2)
            metrics.CIRCUIT_BREAKER_STATE.labels(provider_id=pid).set(state_val)

            if cb.state != CircuitState.OPEN:
                healthy_providers.append(provider)

        if not healthy_providers:
            metrics.GATEWAY_REQUESTS_TOTAL.labels(status="503", tenant_id=tenant).inc()
            raise HTTPException(
                status_code=503,
                detail="All configured LLM providers are currently degraded or have tripped circuit breakers.",
            )

        # Primary selection via active router strategy
        primary_provider = await self.router.select_provider(healthy_providers, request)

        # -------------------------------------------------------------
        # STEP 4: Provider Execution with Failover Cascade
        # -------------------------------------------------------------
        execution_order = [primary_provider]
        for f_id in self.failover_order:
            if f_id in self.providers and self.providers[f_id] != primary_provider:
                if self.circuit_breakers[f_id].state != CircuitState.OPEN:
                    execution_order.append(self.providers[f_id])

        last_error = None
        for idx, provider in enumerate(execution_order):
            try:
                if idx > 0:
                    prev_provider = execution_order[idx - 1]
                    logger.warning(
                        f"Failover triggered: {prev_provider.provider_id} -> {provider.provider_id}"
                    )
                    metrics.CIRCUIT_BREAKER_FAILOVERS_TOTAL.labels(
                        failed_provider_id=prev_provider.provider_id,
                        fallback_provider_id=provider.provider_id,
                    ).inc()

                response = await self._execute_provider_with_circuit_breaker(provider, request)

                # Store into cache asynchronously
                if not request.cache_bypass and self.config.cache.strategy != "disabled":
                    asyncio.create_task(
                        self.cache.set(prompt_text, response, ttl_seconds=self.config.cache.ttl_seconds)
                    )

                total_duration = time.monotonic() - pipeline_start
                response.latency_ms = round(total_duration * 1000.0, 2)
                response.cache_hit = False

                metrics.GATEWAY_REQUESTS_TOTAL.labels(status="200", tenant_id=tenant).inc()
                metrics.GATEWAY_REQUEST_DURATION_SECONDS.labels(
                    provider_id=provider.provider_id, cache_hit="false"
                ).observe(total_duration)
                metrics.TOKENS_PROCESSED_TOTAL.labels(
                    provider_id=provider.provider_id, token_type="input"
                ).inc(response.usage.prompt_tokens)
                metrics.TOKENS_PROCESSED_TOTAL.labels(
                    provider_id=provider.provider_id, token_type="output"
                ).inc(response.usage.completion_tokens)

                return response

            except Exception as e:
                last_error = e
                logger.error(
                    f"Error executing provider '{provider.provider_id}': {e}. Trying next fallback..."
                )
                continue

        metrics.GATEWAY_REQUESTS_TOTAL.labels(status="502", tenant_id=tenant).inc()
        raise HTTPException(
            status_code=502,
            detail=f"All provider attempts and failovers failed. Last error: {str(last_error)}",
        )
