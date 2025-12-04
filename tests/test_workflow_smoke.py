import json
from typing import Any, Dict, List

import pytest

from hybrid_rag_workflow import production_workflow
from hybrid_rag_workflow import workflow_nodes


class StubRetriever:
    """Stub hybrid retriever returning a single synthetic document."""

    def retrieve_from_elasticsearch(
        self,
        query: str,
        correlation_id: str,
        filters: Dict[str, Any] | None = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "content": f"ES content about {query}",
                "title": "ES Innovation",
                "url": "https://example.com/es",
                "score": 0.9,
                "metadata": {"source": "es"},
            }
        ]

    def retrieve_from_chroma(
        self,
        query: str,
        correlation_id: str,
        max_results: int | None = None,
        where_filter: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "content": f"Chroma content about {query}",
                "similarity": 0.95,
                "distance": 0.1,
                "metadata": {"source": "chroma"},
            }
        ]


class StubLLMClient:
    """Stub LLM client that returns fixed JSON for InnovationData."""

    def invoke_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        correlation_id: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "title": "AI-powered renewable energy optimization system",
            "description": "Optimizes wind and solar farm performance using AI.",
            "organization": "Example Energy Corp",
            "sector": "energy",
            "development_stage": "production",
            "funding_amount": 5000000,
            "url": "https://example.com/innovation",
            "confidence_score": 0.92,
        }
        return {
            "content": json.dumps(payload),
            "tokens_used": 123,
            "latency": 0.123,
            "metadata": metadata or {},
        }


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """Patch retrieval and LLM client to avoid real network calls."""
    monkeypatch.setattr(workflow_nodes, "hybrid_retriever", StubRetriever())
    monkeypatch.setattr(workflow_nodes, "llm_client", StubLLMClient())
    yield


def test_workflow_smoke_innovation_extraction():
    query = {"innovation": "AI-powered renewable energy optimization system"}
    flags = {"HTML_flag": True, "org_flag": False}

    result = production_workflow.execute(
        query=query,
        flags=flags,
        thread_id="test-thread-1",
    )

    # `innovation_data` should be a Pydantic model
    innovation = result.get("innovation_data")
    assert innovation is not None
    assert innovation.title.startswith("AI-powered renewable energy")
    assert innovation.organization == "Example Energy Corp"
    assert innovation.sector == "energy"

    metadata = result.get("execution_metadata")
    assert metadata is not None
    assert "duration_seconds" in metadata
    assert "nodes_executed" in metadata
    assert isinstance(metadata["nodes_executed"], list)
    assert "validate_source" in metadata["nodes_executed"]
    assert "extract_innovation" in metadata["nodes_executed"]
