"""Azure OpenAI LLM client with retries, telemetry, and error handling."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from requests.exceptions import ConnectionError, HTTPError, Timeout
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import azure_config
from .telemetry import logger, trace_llm_call, workflow_metrics


class RateLimitError(Exception):
    """Raised when Azure OpenAI rate limit is hit."""


class LLMClient:
    """Production-ready Azure OpenAI client with monitoring and resilience."""

    def __init__(self) -> None:
        self.llm = AzureChatOpenAI(
            azure_deployment=azure_config.deployment_name,
            api_version=azure_config.api_version,
            azure_endpoint=azure_config.endpoint,
            api_key=azure_config.api_key,
            temperature=0.0,  # Deterministic for extraction tasks
            max_retries=0,  # Handle retries manually for better control
            timeout=azure_config.timeout,
        )
        self.deployment_name = azure_config.deployment_name

    @retry(
        stop=stop_after_attempt(azure_config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((HTTPError, Timeout, ConnectionError, RateLimitError)),
        reraise=True,
    )
    def invoke_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        correlation_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke LLM with retry logic and comprehensive telemetry."""
        with trace_llm_call(self.deployment_name, correlation_id) as span:
            start_time = time.time()

            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]

                estimated_input_tokens = len(system_prompt.split()) + len(user_prompt.split())
                span.add_attribute("llm.estimated_input_tokens", estimated_input_tokens)

                response = self.llm.invoke(messages)
                latency = time.time() - start_time

                tokens_used = 0
                if hasattr(response, "response_metadata"):
                    token_usage = response.response_metadata.get("token_usage", {})
                    tokens_used = token_usage.get("total_tokens", 0)
                    span.add_attribute("llm.prompt_tokens", token_usage.get("prompt_tokens", 0))
                    span.add_attribute("llm.completion_tokens", token_usage.get("completion_tokens", 0))

                workflow_metrics.record_llm_call(
                    deployment=self.deployment_name,
                    tokens_used=tokens_used,
                    latency=latency,
                    status_code=200,
                )

                return {
                    "content": response.content,
                    "tokens_used": tokens_used,
                    "latency": latency,
                    "metadata": metadata or {},
                }

            except HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    logger.warning(
                        "Azure OpenAI rate limit hit",
                        extra={
                            "custom_dimensions": {
                                "deployment": self.deployment_name,
                                "correlation_id": correlation_id,
                            }
                        },
                    )
                    workflow_metrics.record_llm_call(
                        deployment=self.deployment_name,
                        tokens_used=0,
                        latency=time.time() - start_time,
                        status_code=429,
                    )
                    raise RateLimitError("Azure OpenAI rate limit exceeded") from exc
                raise

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "LLM invocation failed",
                    extra={
                        "custom_dimensions": {
                            "deployment": self.deployment_name,
                            "correlation_id": correlation_id,
                            "error": str(exc),
                        }
                    },
                    exc_info=True,
                )
                raise


llm_client = LLMClient()
