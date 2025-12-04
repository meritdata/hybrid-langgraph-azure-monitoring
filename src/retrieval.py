"""Hybrid retrieval layer combining Elasticsearch and Chroma."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from elasticsearch import Elasticsearch
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import es_config, vector_db_config
from .telemetry import logger, tracer, workflow_metrics


class HybridRetriever:
    """Production retrieval combining Elasticsearch and Chroma with observability."""

    def __init__(self) -> None:
        # Elasticsearch client for on-premises full-text search
        self.es_client = Elasticsearch(
            hosts=es_config.hosts,
            basic_auth=(es_config.username, es_config.password) if es_config.username else None,
            verify_certs=es_config.verify_certs,
            request_timeout=es_config.timeout,
        )

        # Chroma client for semantic vector search
        self.chroma_client = chromadb.HttpClient(
            host=vector_db_config.host,
            port=vector_db_config.port,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )
        self.collection = self.chroma_client.get_collection(
            name=vector_db_config.collection_name,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def retrieve_from_elasticsearch(
        self,
        query: str,
        correlation_id: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents from on-premises Elasticsearch with telemetry."""
        with tracer.span(name="retrieval.elasticsearch") as span:
            span.add_attribute("correlation_id", correlation_id)
            span.add_attribute("query_length", len(query))

            start_time = time.time()

            try:
                es_query: Dict[str, Any] = {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["title^3", "description^2", "content"],
                                        "type": "best_fields",
                                        "fuzziness": "AUTO",
                                    }
                                }
                            ]
                        }
                    },
                    "size": max_results,
                    "_source": ["title", "description", "url", "sector", "timestamp"],
                }

                if filters:
                    es_query["query"]["bool"]["filter"] = [{"term": {k: v}} for k, v in filters.items()]

                response = self.es_client.search(
                    index=es_config.index_name,
                    body=es_query,
                )

                latency = time.time() - start_time
                hits = response["hits"]["hits"]
                num_results = len(hits)

                documents: List[Dict[str, Any]] = []
                for hit in hits:
                    source = hit["_source"]
                    documents.append(
                        {
                            "content": source.get("description", ""),
                            "title": source.get("title", ""),
                            "url": source.get("url", ""),
                            "score": hit.get("_score", 0.0),
                            "metadata": source,
                        }
                    )

                avg_similarity = (
                    sum(d["score"] for d in documents) / num_results if num_results > 0 else 0.0
                )
                workflow_metrics.record_retrieval(
                    source="elasticsearch",
                    query_latency=latency,
                    num_results=num_results,
                    avg_similarity=avg_similarity,
                )

                span.add_attribute("results_count", num_results)
                span.add_attribute("latency_ms", latency * 1000)

                logger.info(
                    "Elasticsearch retrieval completed",
                    extra={
                        "custom_dimensions": {
                            "correlation_id": correlation_id,
                            "num_results": num_results,
                            "latency_ms": latency * 1000,
                            "query": query[:100],
                        }
                    },
                )

                return documents

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Elasticsearch retrieval failed",
                    extra={
                        "custom_dimensions": {
                            "correlation_id": correlation_id,
                            "error": str(exc),
                            "query": query[:100],
                        }
                    },
                    exc_info=True,
                )
                # Return empty list on failure to allow graceful degradation
                return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def retrieve_from_chroma(
        self,
        query: str,
        correlation_id: str,
        max_results: Optional[int] = None,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve semantically similar documents from Chroma with telemetry."""
        with tracer.span(name="retrieval.chroma") as span:
            span.add_attribute("correlation_id", correlation_id)
            span.add_attribute("query_length", len(query))

            start_time = time.time()

            try:
                n_results = max_results or vector_db_config.max_results

                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )

                latency = time.time() - start_time

                documents: List[Dict[str, Any]] = []
                docs = results.get("documents") or []
                distances = results.get("distances") or []
                metadatas = results.get("metadatas") or []

                if docs and len(docs[0]) > 0:
                    for i, doc in enumerate(docs[0]):
                        distance = distances[0][i]
                        similarity = 1 / (1 + distance)

                        if similarity >= vector_db_config.similarity_threshold:
                            documents.append(
                                {
                                    "content": doc,
                                    "similarity": similarity,
                                    "distance": distance,
                                    "metadata": metadatas[0][i] if metadatas else {},
                                }
                            )

                num_results = len(documents)
                avg_similarity = (
                    sum(d["similarity"] for d in documents) / num_results if num_results > 0 else 0.0
                )

                workflow_metrics.record_retrieval(
                    source="chroma",
                    query_latency=latency,
                    num_results=num_results,
                    avg_similarity=avg_similarity,
                )

                span.add_attribute("results_count", num_results)
                span.add_attribute("latency_ms", latency * 1000)
                span.add_attribute("avg_similarity", avg_similarity)

                logger.info(
                    "Chroma retrieval completed",
                    extra={
                        "custom_dimensions": {
                            "correlation_id": correlation_id,
                            "num_results": num_results,
                            "latency_ms": latency * 1000,
                            "avg_similarity": avg_similarity,
                            "query": query[:100],
                        }
                    },
                )

                if num_results > 0 and avg_similarity < 0.6:
                    logger.warning(
                        "Low semantic similarity in Chroma results",
                        extra={
                            "custom_dimensions": {
                                "correlation_id": correlation_id,
                                "avg_similarity": avg_similarity,
                                "threshold": vector_db_config.similarity_threshold,
                            }
                        },
                    )

                return documents

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Chroma retrieval failed",
                    extra={
                        "custom_dimensions": {
                            "correlation_id": correlation_id,
                            "error": str(exc),
                            "query": query[:100],
                        }
                    },
                    exc_info=True,
                )
                return []


hybrid_retriever = HybridRetriever()
