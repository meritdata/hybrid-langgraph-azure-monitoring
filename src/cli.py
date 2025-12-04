"""Command-line entry point for running the workflow examples."""

from __future__ import annotations

import sys

from .telemetry import logger
from .workflow import production_workflow


def main() -> int:
    """Example production usage with comprehensive error handling."""
    # Example 1: Innovation extraction with HTML source
    query_1 = {
        "query": {"innovation": "AI-powered renewable energy optimization system"},
        "flags": {"HTML_flag": True, "org_flag": False},
    }

    try:
        logger.info(
            "Starting workflow execution",
            extra={"custom_dimensions": {"example": "innovation_extraction"}},
        )

        result = production_workflow.execute(
            query=query_1["query"],
            flags=query_1["flags"],
            thread_id="example-thread-1",
        )

        innovation = result.get("innovation_data")
        if innovation:
            print(f"Extracted Innovation: {innovation.title}")
            print(f"Organization: {innovation.organization}")
            print(f"Confidence: {innovation.confidence_score}")

        validation_issues = [v for v in result.get("validation_results", []) if not v.is_valid]
        if validation_issues:
            logger.warning(
                "Validation issues detected",
                extra={
                    "custom_dimensions": {
                        "issues": [v.model_dump() for v in validation_issues],
                    }
                },
            )

        metadata = result.get("execution_metadata", {})
        print(f"Execution time: {metadata.get('duration_seconds', 0):.2f}s")
        print(f"Nodes executed: {metadata.get('nodes_executed', [])}")

    except Exception:  # noqa: BLE001
        logger.error("Workflow execution failed", exc_info=True)
        return 1

    # Example 2: Direct organization extraction
    query_2 = {
        "query": {"organization": "Tesla Energy Division"},
        "flags": {"HTML_flag": False, "org_flag": True},
    }

    try:
        result = production_workflow.execute(
            query=query_2["query"],
            flags=query_2["flags"],
            thread_id="example-thread-2",
        )

        org = result.get("org_data")
        if org:
            print(f"Organization: {org.name}")
            print(f"Location: {org.location}")

    except Exception:  # noqa: BLE001
        logger.error("Organization extraction failed", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
