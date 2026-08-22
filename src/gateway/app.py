"""
FastAPI Gateway Application Server.
Exposes OpenAI-compatible REST endpoints, Prometheus telemetry, and dynamic algorithm management.
"""

import uuid
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse, Response as RawResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.gateway.models import ChatRequest, ChatResponse
from src.gateway.config import AppConfig
from src.gateway.pipeline import GatewayPipeline
from src.circuit_breaker.states import CircuitState
from src.observability.logger import setup_logger, correlation_id_ctx
from src.observability import metrics

logger = setup_logger("INFO")

# Global pipeline instance
_pipeline: Optional[GatewayPipeline] = None
_config: Optional[AppConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _config
    logger.info("Initializing LLM Gateway...")
    _config = AppConfig.load_from_yaml()
    _pipeline = GatewayPipeline(_config)
    logger.info(
        f"Gateway initialized successfully. RateLimiter={_config.rate_limiter.strategy}, "
        f"Cache={_config.cache.strategy}, Router={_config.router.strategy}"
    )
    yield
    logger.info("Shutting down LLM Gateway...")


app = FastAPI(
    title="Production-Grade LLM Gateway",
    description="High-performance async LLM Gateway with hand-rolled LLD/HLD algorithms (Rate Limiter, Semantic Cache, Circuit Breaker, Router).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_and_timing_middleware(request: Request, call_next):
    # Extract or generate distributed correlation ID
    cid = request.headers.get("X-Correlation-ID", f"req-{uuid.uuid4().hex[:10]}")
    token = correlation_id_ctx.set(cid)

    start_time = time.monotonic()
    try:
        response: Response = await call_next(request)
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        response.headers["X-Correlation-ID"] = cid
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response
    finally:
        correlation_id_ctx.reset(token)


def get_pipeline() -> GatewayPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Gateway pipeline not initialized")
    return _pipeline


# ====================================================================
# INTERACTIVE VISUAL PLAYGROUND
# ====================================================================

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serves the single-page visual control plane & interactive playground."""
    static_file = Path(__file__).resolve().parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file, media_type="text/html")
    return {"message": "LLM Gateway API running. Open /docs for Swagger UI."}


# ====================================================================
# OPENAI-COMPATIBLE API ENDPOINTS
# ====================================================================

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    pipeline: GatewayPipeline = Depends(get_pipeline),
):
    """
    Standard OpenAI-compatible Chat Completions endpoint.
    Routes through Rate Limiter -> Semantic Cache -> Router -> Circuit Breaker -> Provider Adapters.
    """
    return await pipeline.process_chat_completion(request)


@app.get("/v1/models")
async def list_models(pipeline: GatewayPipeline = Depends(get_pipeline)):
    """Returns all enabled LLM provider models."""
    models = []
    for pid, provider in pipeline.providers.items():
        models.append({
            "id": provider.model_name,
            "object": "model",
            "provider_id": pid,
            "owned_by": provider.config.type,
        })
    return {"object": "list", "data": models}


# ====================================================================
# OBSERVABILITY & HEALTH ENDPOINTS
# ====================================================================

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus telemetry scrape endpoint."""
    content, content_type = metrics.get_metrics_output()
    return RawResponse(content=content, media_type=content_type)


@app.get("/health")
async def health_check(pipeline: GatewayPipeline = Depends(get_pipeline)):
    """System health check including status of all downstream provider circuit breakers."""
    cb_states = {
        pid: cb.state.value for pid, cb in pipeline.circuit_breakers.items()
    }
    return {
        "status": "HEALTHY",
        "active_rate_limiter": pipeline.config.rate_limiter.strategy,
        "active_cache": pipeline.config.cache.strategy,
        "cache_entries_count": pipeline.cache.size(),
        "active_router": pipeline.config.router.strategy,
        "circuit_breaker_states": cb_states,
        "providers_configured": list(pipeline.providers.keys()),
    }


# ====================================================================
# GATEWAY CONTROL PLANE & ALGORITHM SWITCHING
# ====================================================================

@app.post("/gateway/cache/clear")
async def clear_cache(pipeline: GatewayPipeline = Depends(get_pipeline)):
    """Purge all entries from the cache."""
    await pipeline.cache.clear()
    return {"status": "SUCCESS", "message": "Cache cleared successfully"}


@app.post("/gateway/circuit-breaker/trip")
async def trip_circuit_breaker(
    provider_id: str = "mock-fast",
    pipeline: GatewayPipeline = Depends(get_pipeline),
):
    """Deliberately trip a provider's circuit breaker to OPEN for outage testing."""
    if provider_id not in pipeline.circuit_breakers:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")
    cb = pipeline.circuit_breakers[provider_id]
    for _ in range(25):
        await cb.record_failure()
    return {
        "status": "SUCCESS",
        "provider_id": provider_id,
        "new_state": cb.state.value,
        "message": f"Tripped '{provider_id}' circuit breaker to {cb.state.value}. Requests will now automatically failover.",
    }


@app.post("/gateway/circuit-breaker/reset")
async def reset_circuit_breakers(pipeline: GatewayPipeline = Depends(get_pipeline)):
    """Reset all circuit breakers back to CLOSED."""
    for pid, cb in pipeline.circuit_breakers.items():
        cb._state = CircuitState.CLOSED
        cb._consecutive_failures = 0
        cb._history.clear()
    return {"status": "SUCCESS", "message": "All circuit breakers reset to CLOSED"}


@app.post("/gateway/reconfigure")
async def reconfigure_gateway(payload: Dict[str, Any]):
    """
    Dynamically update gateway algorithms at runtime from dropdown selections.
    """
    global _pipeline, _config
    if _config is None:
        _config = AppConfig.load_from_yaml()

    # Apply partial updates
    if "rate_limiter_strategy" in payload:
        _config.rate_limiter.strategy = payload["rate_limiter_strategy"]
    if "cache_strategy" in payload:
        _config.cache.strategy = payload["cache_strategy"]
    if "router_strategy" in payload:
        _config.router.strategy = payload["router_strategy"]
    if "circuit_breaker_strategy" in payload:
        _config.circuit_breaker.strategy = payload["circuit_breaker_strategy"]
    if "similarity_threshold" in payload:
        _config.cache.similarity_threshold = float(payload["similarity_threshold"])

    _pipeline = GatewayPipeline(_config)
    logger.info(f"Gateway dynamically reconfigured: RateLimiter={_config.rate_limiter.strategy}, Cache={_config.cache.strategy}, Router={_config.router.strategy}")
    return {
        "status": "SUCCESS",
        "active_rate_limiter": _config.rate_limiter.strategy,
        "active_cache": _config.cache.strategy,
        "active_router": _config.router.strategy,
        "active_circuit_breaker": _config.circuit_breaker.strategy,
        "similarity_threshold": _config.cache.similarity_threshold,
    }


if __name__ == "__main__":
    import uvicorn
    cfg = AppConfig.load_from_yaml()
    uvicorn.run(
        "src.gateway.app:app",
        host=cfg.gateway.host,
        port=cfg.gateway.port,
        reload=False,
    )
