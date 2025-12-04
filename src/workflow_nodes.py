"""LangGraph workflow nodes implementing validation, retrieval, and extraction."""

from __future__ import annotations

import json
from typing import Any, Dict

from pydantic import ValidationError

from .llm_client import llm_client
from .retrieval import hybrid_retriever
from .state import InnovationData, OrganizationData, State, ValidationResult
from .telemetry import logger, trace_node, workflow_metrics


@trace_node("validate_source")
def validate_source(state: State) -> Dict[str, Any]:
    """Validate input source and set workflow routing flags."""
    correlation_id = state["correlation_id"]
    query = state.get("query", {})

    html_flag = state["flags"].get("HTML_flag", False)
    org_flag = state["flags"].get("org_flag", False)

    if not query or not query.get("innovation"):
        logger.error(
            "Invalid query structure",
            extra={
                "custom_dimensions": {
                    "correlation_id": correlation_id,
                    "query": query,
                }
            },
        )
        state["errors"].append(
            {"node": "validate_source", "message": "Missing innovation query"}
        )
        return {
            "flags": {"error": True},
            "errors": state["errors"],
        }

    state["node_execution_history"].append("validate_source")

    return {
        "flags": {"HTML_flag": html_flag, "org_flag": org_flag, "validated": True},
        "node_execution_history": state["node_execution_history"],
    }


@trace_node("retrieve_innovation")
def retrieve_innovation_docs(state: State) -> Dict[str, Any]:
    """Retrieve relevant innovation documents using hybrid search strategy."""
    correlation_id = state["correlation_id"]
    query_text = state["query"].get("innovation", "")

    es_docs = hybrid_retriever.retrieve_from_elasticsearch(
        query=query_text,
        correlation_id=correlation_id,
        max_results=10,
    )

    chroma_docs = hybrid_retriever.retrieve_from_chroma(
        query=query_text,
        correlation_id=correlation_id,
        max_results=5,
    )

    all_docs = es_docs + chroma_docs

    if not all_docs:
        logger.warning(
            "No documents retrieved for innovation query",
            extra={
                "custom_dimensions": {
                    "correlation_id": correlation_id,
                    "query": query_text[:100],
                }
            },
        )
        workflow_metrics.record_validation_failure(
            field_name="document_retrieval",
            error_type="no_results",
            value=query_text,
        )

    state["node_execution_history"].append("retrieve_innovation")

    return {
        "documents": [doc.get("content", "") for doc in all_docs],
        "elasticsearch_results": es_docs,
        "chroma_results": chroma_docs,
        "node_execution_history": state["node_execution_history"],
    }


@trace_node("extract_innovation")
def extract_innovation_data(state: State) -> Dict[str, Any]:
    """Extract structured innovation data using LLM with RAG context."""
    correlation_id = state["correlation_id"]
    query_text = state["query"].get("innovation", "")
    documents = state.get("documents", [])

    context = "\n\n".join([f"Document {i + 1}:\n{doc}" for i, doc in enumerate(documents[:5])])

    if not context:
        context = "No relevant context documents available."
        logger.warning(
            "Extracting innovation without context",
            extra={"custom_dimensions": {"correlation_id": correlation_id}},
        )

    system_prompt = """You are an expert at extracting structured innovation data from unstructured text.
Extract the following fields in valid JSON format:
- title (string, required): Name of the innovation
- description (string): Detailed description
- organization (string): Organization developing the innovation
- sector (string): Industry sector (technology/healthcare/energy/finance/other)
- development_stage (string): Current stage (research/prototype/production/other)
- funding_amount (number): Funding in USD if mentioned
- url (string): Relevant URL if available

Return ONLY valid JSON with these fields. Use null for missing values."""

    user_prompt = f"""Query: {query_text}

Context Documents:
{context}

Extract innovation data as JSON:"""

    try:
        llm_response = llm_client.invoke_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            correlation_id=correlation_id,
            metadata={"query": query_text},
        )

        content = llm_response["content"]

        try:
            parsed_data = json.loads(content)
            innovation = InnovationData(**parsed_data)

            state["node_execution_history"].append("extract_innovation")

            return {
                "innovation_data": innovation,
                "node_execution_history": state["node_execution_history"],
            }

        except json.JSONDecodeError as exc:
            logger.error(
                "LLM returned invalid JSON",
                extra={
                    "custom_dimensions": {
                        "correlation_id": correlation_id,
                        "llm_response": content[:500],
                        "error": str(exc),
                    }
                },
            )
            workflow_metrics.record_validation_failure(
                field_name="llm_output",
                error_type="invalid_json",
                value=content[:200],
            )
            raise

        except ValidationError as exc:
            logger.error(
                "Innovation data validation failed",
                extra={
                    "custom_dimensions": {
                        "correlation_id": correlation_id,
                        "validation_errors": exc.errors(),
                        "data": parsed_data,
                    }
                },
            )
            for error in exc.errors():
                workflow_metrics.record_validation_failure(
                    field_name=".".join(str(loc) for loc in error["loc"]),
                    error_type=error["type"],
                    value=error.get("input"),
                )
            raise

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Innovation extraction failed",
            extra={
                "custom_dimensions": {
                    "correlation_id": correlation_id,
                    "error": str(exc),
                }
            },
            exc_info=True,
        )
        state["errors"].append(
            {
                "node": "extract_innovation",
                "error": str(exc),
                "query": query_text,
            }
        )
        return {"errors": state["errors"]}


@trace_node("pipeline_decision")
def pipeline_decision(state: State) -> str:
    """Determine workflow routing based on flags and state."""
    html_flag = state["flags"].get("HTML_flag", False)
    org_flag = state["flags"].get("org_flag", False)
    error_flag = state["flags"].get("error", False)

    if error_flag:
        return "error_handler"

    if html_flag and org_flag:
        return "innovation_only"
    if html_flag:
        return "innovation_plus_org"
    if org_flag:
        return "direct_org_wrapper"
    return "open_search_mode"


@trace_node("validate_results")
def validate_results(state: State) -> Dict[str, Any]:
    """Validate extracted data quality and completeness."""
    correlation_id = state["correlation_id"]
    validation_results: list[ValidationResult] = []

    innovation_data = state.get("innovation_data")

    if innovation_data:
        if not innovation_data.title:
            validation_results.append(
                ValidationResult(
                    field_name="title",
                    is_valid=False,
                    error_type="missing_required",
                    error_message="Title is required but missing",
                )
            )
            workflow_metrics.record_validation_failure("title", "missing_required")

        if innovation_data.confidence_score is not None and innovation_data.confidence_score < 0.5:
            validation_results.append(
                ValidationResult(
                    field_name="confidence_score",
                    is_valid=False,
                    error_type="low_confidence",
                    error_message=(
                        f"Confidence score {innovation_data.confidence_score} below threshold"
                    ),
                )
            )
            logger.warning(
                "Low confidence in extraction",
                extra={
                    "custom_dimensions": {
                        "correlation_id": correlation_id,
                        "confidence": innovation_data.confidence_score,
                    }
                },
            )

    state["node_execution_history"].append("validate_results")

    return {
        "validation_results": validation_results,
        "node_execution_history": state["node_execution_history"],
    }
