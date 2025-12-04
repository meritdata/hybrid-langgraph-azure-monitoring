"""LangGraph workflow construction and execution."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, START, StateGraph

from .config import workflow_config
from .state import State, create_initial_state
from .telemetry import logger, tracer
from .workflow_nodes import (
    extract_innovation_data,
    pipeline_decision,
    retrieve_innovation_docs,
    validate_results,
    validate_source,
)


class ProductionWorkflow:
    """Production-ready LangGraph workflow with checkpointing and monitoring."""

    def __init__(self) -> None:
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph workflow with all nodes and edges."""
        builder = StateGraph(State)

        builder.add_node("validate_source", validate_source)
        builder.add_node("retrieve_innovation", retrieve_innovation_docs)
        builder.add_node("extract_innovation", extract_innovation_data)
        builder.add_node("validate_results", validate_results)
        builder.add_node("final_wrapper", self._wrap_output)
        builder.add_node("error_handler", self._handle_error)

        builder.add_edge(START, "validate_source")

        builder.add_conditional_edges(
            "validate_source",
            pipeline_decision,
            {
                "innovation_only": "retrieve_innovation",
                "innovation_plus_org": "retrieve_innovation",
                "direct_org_wrapper": "final_wrapper",
                "open_search_mode": "retrieve_innovation",
                "error_handler": "error_handler",
            },
        )

        builder.add_edge("retrieve_innovation", "extract_innovation")
        builder.add_edge("extract_innovation", "validate_results")
        builder.add_edge("validate_results", "final_wrapper")

        builder.add_edge("error_handler", END)
        builder.add_edge("final_wrapper", END)

        if workflow_config.enable_checkpointing:
            if workflow_config.checkpoint_backend == "redis" and workflow_config.redis_url:
                checkpointer = RedisSaver.from_conn_string(workflow_config.redis_url)
            else:
                checkpointer = MemorySaver()
            return builder.compile(checkpointer=checkpointer)

        return builder.compile()

    @staticmethod
    def _wrap_output(state: State) -> Dict[str, Any]:
        """Prepare final output metadata and attach it to the state.

        The full state (including innovation_data, org_data, validation_results, etc.)
        is returned by LangGraph. This node only adds the execution_metadata block.
        """
        execution_time = time.time() - state["execution_start_time"]

        execution_metadata = {
            "duration_seconds": execution_time,
            "nodes_executed": state["node_execution_history"],
            "retry_count": state.get("retry_count", 0),
            "errors": state.get("errors", []),
        }

        logger.info(
            "Workflow completed successfully",
            extra={
                "custom_dimensions": {
                    "correlation_id": state["correlation_id"],
                    "duration_seconds": execution_time,
                    "nodes_executed": len(state["node_execution_history"]),
                }
            },
        )

        state["node_execution_history"].append("final_wrapper")

        return {
            "execution_metadata": execution_metadata,
            "node_execution_history": state["node_execution_history"],
        }

    @staticmethod
    def _handle_error(state: State) -> Dict[str, Any]:
        """Handle workflow errors with logging and alerting."""
        errors = state.get("errors", [])

        logger.error(
            "Workflow failed with errors",
            extra={
                "custom_dimensions": {
                    "correlation_id": state["correlation_id"],
                    "errors": errors,
                    "nodes_executed": state["node_execution_history"],
                }
            },
        )

        state["node_execution_history"].append("error_handler")
        return {"node_execution_history": state["node_execution_history"]}

    def execute(
        self,
        query: Dict[str, str],
        flags: Dict[str, bool],
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute workflow with comprehensive error handling and tracing."""
        initial_state = create_initial_state(query=query, flags=flags)
        correlation_id = initial_state["correlation_id"]

        with tracer.span(name="workflow.execution") as span:
            span.add_attribute("correlation_id", correlation_id)
            span.add_attribute("query", str(query)[:100])

            try:
                config: Dict[str, Any] = {}
                if thread_id:
                    config = {"configurable": {"thread_id": thread_id}}

                result = self.graph.invoke(initial_state, config=config)

                span.add_attribute("success", True)
                span.add_attribute(
                    "execution_time",
                    time.time() - initial_state["execution_start_time"],
                )

                return result

            except Exception as exc:  # noqa: BLE001
                span.add_attribute("success", False)
                span.add_attribute("error", str(exc))

                logger.error(
                    "Workflow execution failed",
                    extra={
                        "custom_dimensions": {
                            "correlation_id": correlation_id,
                            "error": str(exc),
                            "query": query,
                        }
                    },
                    exc_info=True,
                )
                raise


production_workflow = ProductionWorkflow()
