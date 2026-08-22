"""
Locust Load Testing Harness for LLM Gateway.

Simulates realistic multi-tenant production traffic:
1. Exact Repeat Queries (Evaluates Exact Cache Hit rate & 0ms cost)
2. Paraphrased Queries (Evaluates Semantic Cache Hit rate)
3. New Unique Queries (Evaluates Router, Circuit Breakers, and Provider Latency)
4. Burst Traffic (Evaluates Rate Limiter throttling & 429 Retry-After responses)
"""

import random
from locust import HttpUser, task, between


SAMPLE_PROMPTS = [
    "Explain how Consistent Hashing works in distributed systems.",
    "What is the difference between Token Bucket and Leaky Bucket algorithms?",
    "How does a Circuit Breaker prevent cascading failure in microservices?",
    "Write a Python function to implement LRU cache using Doubly Linked List.",
    "What is the time complexity of vector cosine similarity with normalized embeddings?",
]

PARAPHRASED_PROMPTS = [
    "Can you explain how consistent hash rings operate in distributed caching?",
    "Compare token bucket vs leaky bucket rate limiting mechanisms.",
    "Explain circuit breaker state machine CLOSED OPEN HALF_OPEN.",
    "Show me python code for LRU cache with hash map and doubly linked list.",
    "What is the Big O complexity of cosine similarity dot product?",
]


class LLMGatewayUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.tenant_id = f"tenant_{random.randint(1, 5)}"

    @task(4)
    def test_repeated_queries(self):
        """Tests Cache Hit rate."""
        prompt = random.choice(SAMPLE_PROMPTS)
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "tenant_id": self.tenant_id,
        }
        self.client.post("/v1/chat/completions", json=payload)

    @task(3)
    def test_paraphrased_queries(self):
        """Tests Semantic Vector Cache."""
        prompt = random.choice(PARAPHRASED_PROMPTS)
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "tenant_id": self.tenant_id,
        }
        self.client.post("/v1/chat/completions", json=payload)

    @task(2)
    def test_unique_queries(self):
        """Tests Cache Miss -> Router -> Provider."""
        prompt = f"Unique query {random.random()}: What is {random.randint(1, 10000)} squared?"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "tenant_id": self.tenant_id,
        }
        self.client.post("/v1/chat/completions", json=payload)

    @task(1)
    def test_burst_rate_limiting(self):
        """Fires rapid burst to deliberately trigger HTTP 429."""
        burst_tenant = f"burst_user_{random.randint(1, 2)}"
        payload = {
            "messages": [{"role": "user", "content": "Rapid burst message"}],
            "tenant_id": burst_tenant,
        }
        for _ in range(12):
            self.client.post("/v1/chat/completions", json=payload)
