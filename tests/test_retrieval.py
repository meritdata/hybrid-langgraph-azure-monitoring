from typing import Any, Dict, List

from hybrid_rag_workflow.retrieval import HybridRetriever
from hybrid_rag_workflow.telemetry import workflow_metrics


class DummyESClient:
    """Minimal Elasticsearch stub."""

    def search(self, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 1.23,
                        "_source": {
                            "title": "ES Innovation",
                            "description": "ES description",
                            "url": "https://example.com/es",
                            "sector": "energy",
                            "timestamp": "2024-01-01T00:00:00Z",
                        },
                    }
                ]
            }
        }


class DummyChromaCollection:
    """Minimal Chroma collection stub."""

    def query(
        self,
        query_texts: List[str],
        n_results: int,
        where: Dict[str, Any] | None = None,
        include: List[str] | None = None,
    ) -> Dict[str, Any]:
        return {
            "documents": [[f"Chroma doc about {query_texts[0]}"]],
            "distances": [[0.1]],
            "metadatas": [[{"source": "chroma"}]],
        }


def _make_test_retriever() -> HybridRetriever:
    """Create a HybridRetriever instance without running its real __init__."""
    retriever = HybridRetriever.__new__(HybridRetriever)  # type: ignore[misc]
    retriever.es_client = DummyESClient()
    retriever.chroma_client = object()  # not used directly in tests
    retriever.collection = DummyChromaCollection()
    return retriever


def test_retrieve_from_elasticsearch_basic():
    retriever = _make_test_retriever()

    result = retriever.retrieve_from_elasticsearch(
        query="renewable energy",
        correlation_id="test-corr-id",
        max_results=5,
    )

    assert len(result) == 1
    doc = result[0]
    assert doc["title"] == "ES Innovation"
    assert "content" in doc
    assert doc["url"] == "https://example.com/es"


def test_retrieve_from_chroma_similarity_and_filtering():
    retriever = _make_test_retriever()

    result = retriever.retrieve_from_chroma(
        query="renewable energy",
        correlation_id="test-corr-id",
        max_results=3,
    )

    assert len(result) == 1
    doc = result[0]
    assert "similarity" in doc
    assert doc["similarity"] > 0.0
    assert "content" in doc
    assert "distance" in doc


def test_workflow_metrics_record_retrieval_called():
    # This test checks that record_retrieval can be called without errors
    # using reasonable payloads; it does not assert against App Insights.
    workflow_metrics.record_retrieval(
        source="test",
        query_latency=0.05,
        num_results=2,
        avg_similarity=0.8,
    )

    # Basic sanity: internal maps should be dicts (no exceptions thrown).
    assert isinstance(workflow_metrics.node_executions, dict)
