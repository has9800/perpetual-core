"""
Prometheus Metrics Service
Exposes metrics for monitoring and observability
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time

# Request metrics
request_counter = Counter(
    'perpetual_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status_code']
)

request_latency = Histogram(
    'perpetual_request_duration_seconds',
    'Request latency in seconds',
    ['endpoint', 'method'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Token usage metrics
token_counter = Counter(
    'perpetual_tokens_total',
    'Total tokens processed',
    ['type', 'model']  # type: input, output, total
)

# Memory/retrieval metrics
retrieval_counter = Counter(
    'perpetual_retrievals_total',
    'Total memory retrievals',
    ['conversation_id']
)

retrieval_latency = Histogram(
    'perpetual_retrieval_duration_seconds',
    'Memory retrieval latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# Active connections
active_connections = Gauge(
    'perpetual_active_connections',
    'Number of active connections'
)

# Error tracking
error_counter = Counter(
    'perpetual_errors_total',
    'Total errors',
    ['endpoint', 'error_type']
)

# Cost tracking
cost_counter = Counter(
    'perpetual_cost_usd_total',
    'Total cost in USD',
    ['user_id', 'model']
)


class MetricsService:
    """Service for recording Prometheus metrics"""

    @staticmethod
    def record_request(
        endpoint: str,
        method: str,
        status_code: int,
        duration_seconds: float
    ):
        """Record a request"""
        request_counter.labels(
            endpoint=endpoint,
            method=method,
            status_code=status_code
        ).inc()

        request_latency.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration_seconds)

    @staticmethod
    def record_tokens(
        token_type: str,  # 'input', 'output', 'total'
        count: int,
        model: str
    ):
        """Record token usage"""
        token_counter.labels(
            type=token_type,
            model=model
        ).inc(count)

    @staticmethod
    def record_retrieval(
        conversation_id: str,
        duration_seconds: float
    ):
        """Record memory retrieval"""
        retrieval_counter.labels(
            conversation_id=conversation_id
        ).inc()

        retrieval_latency.observe(duration_seconds)

    @staticmethod
    def record_error(
        endpoint: str,
        error_type: str
    ):
        """Record an error"""
        error_counter.labels(
            endpoint=endpoint,
            error_type=error_type
        ).inc()

    @staticmethod
    def record_cost(
        user_id: str,
        model: str,
        cost_usd: float
    ):
        """Record cost"""
        cost_counter.labels(
            user_id=user_id,
            model=model
        ).inc(cost_usd)

    @staticmethod
    def increment_connections():
        """Increment active connections"""
        active_connections.inc()

    @staticmethod
    def decrement_connections():
        """Decrement active connections"""
        active_connections.dec()

    @staticmethod
    def export_metrics() -> Response:
        """Export Prometheus metrics"""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


# Singleton instance
_metrics_service: MetricsService = None


def get_metrics_service() -> MetricsService:
    """Get or create metrics service instance"""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service
