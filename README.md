# Monitoring and Maintaining an AI-Powered Data Extraction Model

_A Technical Deep Dive into Hybrid Azure + On-Premises Deployment with LangGraph, Azure OpenAI, Elasticsearch, and Chroma_

This repository contains the reference implementation that accompanies the blog:

> **“Monitoring LangGraph + Azure OpenAI Extraction at Scale: Hybrid Telemetry, Embedding Drift, and State-Aware Observability.”**

It focuses on the **runtime reality** of an AI-powered data extraction system running across **Azure** and **on-premises** infrastructure: how to monitor it, how to keep it healthy, and how to reason about failures that look like “the model got worse” but are actually caused by **drift**, **integration issues**, or **infrastructure behaviour**.

---

## 1. What this repo is (and isn’t)

This codebase is a **narrow, production-style slice** of the larger architecture described in the blog:

- ✅ Implements a **LangGraph-based extraction workflow** using:
  - Azure OpenAI (via `AzureChatOpenAI`)
  - Hybrid retrieval with **Elasticsearch** (on-prem) + **Chroma** (on-prem)
  - **Azure Application Insights** for telemetry (logs + traces + custom metrics)
  - **Redis or in-memory checkpointing** for LangGraph state
- ✅ Demonstrates **production patterns**:
  - Pydantic-based configuration management
  - Correlation IDs and stateful tracing
  - Node-level metrics and LLM call telemetry
  - Retry & rate-limit handling for Azure OpenAI
  - Hybrid retrieval with graceful degradation

It does **not** try to fully reproduce:

- Apache Airflow DAGs
- SearXNG-based scraping infrastructure
- Power BI dashboards / full BI layer
- Azure ML model lifecycle tooling

Those are treated as architectural context in the blog; this repo zooms in on the **workflow, telemetry, and observability core**.

---

## 2. Architecture Overview

At a high level, this repo implements the **Azure-side orchestration** for a hybrid deployment:

- **On-Premises (conceptual, not fully implemented here)**  
  - Apache Airflow orchestrates scraping and preprocessing.  
  - Elasticsearch stores raw and enriched innovation documents.  
  - Chroma stores embeddings for semantic search.  
  - Data + embeddings are exposed to Azure via a private network (e.g. ExpressRoute).

- **Azure (implemented here)**  
  - **LangGraph** orchestrates the extraction workflow with stateful nodes.  
  - **Azure OpenAI** performs LLM-based extraction (e.g., GPT-4o Mini).  
  - **Redis or in-memory checkpointing** persists workflow state.  
  - **Azure Application Insights** captures logs, traces, and custom metrics.  

The workflow is designed around **correlation IDs**, **state-aware logging**, and **metrics** that let you distinguish:

- “The model is worse” vs “retrieval failed” vs “network jitter” vs “checkpointing problems”.
- Embedding / vector drift vs Airflow SLA issues vs tokenisation mismatches.

---

## 3. Repository Structure

```text
.
├── README.md
├── LICENSE
├── pyproject.toml          # or setup.cfg/setup.py
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yml
├── .env.example
├── src/
│   └── hybrid_rag_workflow/
│       ├── __init__.py
│       ├── config.py           # Pydantic-based config (Azure, Chroma, ES, monitoring, workflow)
│       ├── telemetry.py        # App Insights logger, tracer, custom metrics, decorators
│       ├── state.py            # Typed State + Pydantic models for extracted data and validation
│       ├── llm_client.py       # Azure OpenAI client with retries + telemetry
│       ├── retrieval.py        # Hybrid retrieval layer (Elasticsearch + Chroma)
│       ├── workflow_nodes.py   # LangGraph nodes (validate, retrieve, extract, validate_results)
│       ├── workflow.py         # Graph construction, checkpointing, execution wrapper
│       └── cli.py              # Example CLI entrypoint (was main.py)
├── notebooks/
│   └── 01_innovation_extraction_demo.ipynb
└── tests/
    ├── __init__.py
    ├── test_state.py
    ├── test_workflow_smoke.py
    └── test_retrieval.py
