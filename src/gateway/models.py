"""
Unified data models for chat completions, requests, responses, and token metrics.
Follows the OpenAI-compatible REST API specification format for universal client compatibility.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
import time
import uuid


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"] = "user"
    content: str
    name: Optional[str] = None


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1024, gt=0)
    stream: bool = False
    
    # Gateway specific routing / control overrides
    tenant_id: Optional[str] = "default_tenant"
    preferred_provider: Optional[str] = None
    cache_bypass: bool = False
    session_id: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo
    
    # Telemetry metadata injected by the gateway
    provider_id: Optional[str] = None
    cache_hit: bool = False
    cache_type: Optional[Literal["EXACT", "SEMANTIC", "NONE"]] = "NONE"
    latency_ms: float = 0.0
