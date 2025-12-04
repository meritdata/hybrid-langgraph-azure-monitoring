import re
from datetime import datetime

from hybrid_rag_workflow.state import (
    InnovationData,
    State,
    ValidationResult,
    create_initial_state,
)


def test_create_initial_state_structure():
    query = {"innovation": "test innovation"}
    flags = {"HTML_flag": True, "org_flag": False}

    state = create_initial_state(query=query, flags=flags)

    assert isinstance(state, dict)
    assert state["query"] == query
    assert state["flags"] == flags

    # correlation_id should look like a UUID
    assert re.match(
        r"^[0-9a-fA-F-]{36}$",
        state["correlation_id"],
    )

    assert isinstance(state["execution_start_time"], float)
    assert state["documents"] == []
    assert state["errors"] == []
    assert state["node_execution_history"] == []
    assert state["retry_count"] == 0


def test_innovation_data_sector_normalization():
    data = InnovationData(
        title="Test",
        sector="Technology",
    )
    assert data.sector == "technology"

    data_unknown = InnovationData(
        title="Test",
        sector="unknown-sector",
    )
    assert data_unknown.sector == "other"


def test_validation_result_defaults():
    vr = ValidationResult(
        field_name="title",
        is_valid=False,
        error_type="missing_required",
        error_message="Title is required",
    )

    assert vr.field_name == "title"
    assert not vr.is_valid
    assert vr.error_type == "missing_required"
    assert isinstance(vr.validated_at, datetime)
