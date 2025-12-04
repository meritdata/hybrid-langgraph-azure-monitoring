"""Shared workflow state and validated data models."""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


class InnovationData(BaseModel):
    """Validated innovation extraction output."""

    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=5000)
    organization: Optional[str] = Field(None, max_length=200)
    sector: Optional[str] = None
    development_stage: Optional[str] = None
    funding_amount: Optional[float] = Field(None, ge=0)
    url: Optional[str] = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    _allowed_sectors = {"technology", "healthcare", "energy", "finance", "other"}

    @field_validator("sector")
    @classmethod
    def validate_sector(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        lower = value.lower()
        if lower not in cls._allowed_sectors:
            return "other"
        return lower


class OrganizationData(BaseModel):
    """Validated organization metadata."""

    name: str = Field(..., min_length=1)
    website: Optional[str] = None
    location: Optional[str] = None
    employee_count: Optional[int] = Field(None, ge=0)
    founded_year: Optional[int] = Field(None, ge=1800, le=datetime.now().year)


class ValidationResult(BaseModel):
    """Data quality validation tracking."""

    field_name: str
    is_valid: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    validated_at: datetime = Field(default_factory=datetime.utcnow)


class State(TypedDict):
    """Shared memory of the Web Agent workflow."""

    # Unique identifier for distributed tracing
    correlation_id: str

    # Input query and flags
    query: Dict[str, str]
    flags: Dict[str, bool]

    # Retrieved documents and context
    documents: List[str]
    elasticsearch_results: Optional[List[Dict[str, Any]]]
    chroma_results: Optional[List[Dict[str, Any]]]

    # Extracted structured data
    innovation_data: Optional[InnovationData]
    org_data: Optional[OrganizationData]

    # Validation and quality tracking
    validation_results: List[ValidationResult]

    # Execution metadata
    execution_start_time: float
    node_execution_history: List[str]
    retry_count: int

    # Error tracking
    errors: List[Dict[str, Any]]


def create_initial_state(query: Dict[str, str], flags: Dict[str, bool]) -> State:
    """Factory function to create validated initial state."""
    return State(
        correlation_id=str(uuid.uuid4()),
        query=query,
        flags=flags,
        documents=[],
        elasticsearch_results=None,
        chroma_results=None,
        innovation_data=None,
        org_data=None,
        validation_results=[],
        execution_start_time=time.time(),
        node_execution_history=[],
        retry_count=0,
        errors=[],
    )
