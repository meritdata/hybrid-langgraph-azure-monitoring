"""Telemetry and observability utilities for the workflow."""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict

from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace import config_integration
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.span import SpanKind
from opencensus.trace.tracer import Tracer

from .config import monitoring_config

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------

logger = logging.getLogger("hybrid_rag_workflow")

if not logger.handlers:
    logger.setLevel(monitoring_config.log_level.upper())
    logger.addHandler(
        AzureLogHandler(
            connection_string=f"InstrumentationKey={monitoring_config.instrumentation_key}"
        )
    )

# ------------------------------------------------------------------------------
# Distributed tracing configuration
# ------------------------------------------------------------------------------

if monitoring_config.enable_distributed_tracing:
    config_integration.trace_integrations(["requests", "httplib"])

tracer = Tracer(
    exporter=AzureExporter(
        connection_string=f"InstrumentationKey={monitoring_config.instrumentation_key}"
    ),
    sampler=ProbabilitySampler(rate=monitoring_config.sample_rate),
)


class WorkflowMetrics:
    """Custom metrics for LangGraph workflow monitoring."""

    def __init__(self) -> None:
        self.node_executions: Dict[str, int] = {}
        self.node_failures: Dict[str, int] = {}
        self.node_latencies: Dict[str, list[float]] = {}

    def record_execution(
        self,
        node_name: str,
        success: bool,
        duration: float,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Record node execution metrics."""
        logger.info(
            "Node execution completed",
            extra={
                "custom_dimensions": {
                    "node_name": node_name,
                    "success": success,
                    "duration_ms": duration * 1000,
                    "metadata": metadata or {},
                }
            },
        )

        if node_name not in self.node_executions:
            self.node_executions[node_name] = 0
            self.node_failures[node_name] = 0
            self.node_latencies[node_name] = []

        self.node_executions[node_name] += 1
        if not success:
            self.node_failures[node_name] += 1
        self.node_latencies[node_name].append(duration)

    def record_llm_call(
        self, deployment: str, tokens_used: int, latency: float, status_code: int
    ) -> None:
        """Record LLM API call metrics."""
        logger.info(
            "LLM API call",
            extra={
                "custom_dimensions": {
                    "deployment": deployment,
                    "tokens_used": tokens_used,
                    "latency_ms": latency * 1000,
                    "status_code": status_code,
                    "throttled": status_code == 429,
                }
            },
        )

    def record_retrieval(
        self, source: str, query_latency: float, num_results: int, avg_similarity: float
    ) -> None:
        """Record vector/text retrieval metrics."""
        logger.info(
            "Document retrieval",
            extra={
                "custom_dimensions": {
                    "source": source,
                    "query_latency_ms": query_latency * 1000,
                    "num_results": num_results,
                    "avg_similarity_score": avg_similarity,
                }
            },
        )

    def record_validation_failure(
        self,
        field_name: str,
        error_type: str,
        value: Any | None = None,
    ) -> None:
        """Record data quality validation failures."""
        logger.warning(
            "Validation failure",
            extra={
                "custom_dimensions": {
                    "field_name": field_name,
                    "error_type": error_type,
                    "invalid_value": str(value) if value is not None else None,
                }
            },
        )


workflow_metrics = WorkflowMetrics()


def trace_node(node_name: str) -> Callable[[Callable[..., Dict[str, Any]]], Callable[..., Dict[str, Any]]]:
    """Decorator to add distributed tracing and metrics to workflow nodes."""

    def decorator(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
        @functools.wraps(func)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            with tracer.span(name=f"workflow.node.{node_name}") as span:
                span.span_kind = SpanKind.INTERNAL
                span.add_attribute("node.name", node_name)
                span.add_attribute("correlation_id", state.get("correlation_id", "unknown"))

                start_time = time.time()
                success = True
                error_message: str | None = None

                try:
                    result = func(state)
                    span.add_attribute("node.output_size", len(str(result)))
                    return result
                except Exception as exc:  # noqa: BLE001
                    success = False
                    error_message = str(exc)
                    span.add_attribute("error", True)
                    span.add_attribute("error.message", error_message)
                    logger.error(
                        "Node %s failed", node_name,
                        extra={
                            "custom_dimensions": {
                                "node_name": node_name,
                                "error": error_message,
                                "correlation_id": state.get("correlation_id"),
                            }
                        },
                        exc_info=True,
                    )
                    raise
                finally:
                    duration = time.time() - start_time
                    workflow_metrics.record_execution(
                        node_name=node_name,
                        success=success,
                        duration=duration,
                        metadata={"error": error_message} if error_message else None,
                    )

        return wrapper

    return decorator


@contextmanager
def trace_llm_call(deployment: str, correlation_id: str):
    """Context manager for tracing LLM API calls."""
    with tracer.span(name=f"llm.call.{deployment}") as span:
        span.span_kind = SpanKind.CLIENT
        span.add_attribute("llm.deployment", deployment)
        span.add_attribute("correlation_id", correlation_id)

        start_time = time.time()
        try:
            yield span
        finally:
            latency = time.time() - start_time
            span.add_attribute("llm.latency_ms", latency * 1000)
