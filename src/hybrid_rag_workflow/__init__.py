"""Hybrid LangGraph + Azure OpenAI monitoring demo.

This package implements a production-style LangGraph workflow for
intelligent data extraction, with hybrid retrieval (Elasticsearch + Chroma),
Azure OpenAI inference, and Application Insights telemetry.

See README.md and the accompanying blog for architecture context.
"""

from .workflow import production_workflow  # noqa: F401
