from __future__ import annotations

import copy
import json
import os
from unittest.mock import patch

import pytest

from livekit.agents.beta.gtm_telemetry.adapters import (
    build_hubspot_engagement,
    build_salesforce_task,
)
from livekit.agents.beta.gtm_telemetry.models import PostCallReport, TranscriptTurn

pytestmark = pytest.mark.unit


def _report(*, transcript_text: str | None = None) -> PostCallReport:
    transcript = []
    if transcript_text is not None:
        transcript = [
            TranscriptTurn(
                item_id="item_1",
                role="user",
                text=transcript_text,
                interrupted=False,
                created_at=1.0,
            )
        ]
    return PostCallReport(
        report_id="r1",
        report_created_at=0.0,
        job_id="job-1",
        room_name="room-1",
        started_at=100.0,
        ended=True,
        end_reason="user_initiated",
        duration=42.0,
        transcript=transcript,
        metadata={"disposition": "resolved"},
    )


# --- HubSpot ---------------------------------------------------------------------------


def test_hubspot_engagement_shape_has_required_fields():
    payload = build_hubspot_engagement(_report(), owner_id="owner_1", contact_ids=["c1", "c2"])
    assert payload["engagement"]["type"] == "CALL"
    assert payload["associations"]["ownerId"] == "owner_1"
    assert payload["associations"]["contactIds"] == ["c1", "c2"]
    assert payload["source"]["jobId"] == "job-1"


def test_hubspot_engagement_transcript_truncated_with_metadata():
    report = _report(transcript_text="x" * 100)
    payload = build_hubspot_engagement(report, transcript_max_chars=10)
    assert payload["metadata"]["transcriptTruncated"] is True
    assert payload["metadata"]["transcriptOmittedChars"] > 0
    assert len(payload["metadata"]["body"]) == 10


def test_hubspot_engagement_no_truncation_when_under_limit():
    report = _report(transcript_text="short")
    payload = build_hubspot_engagement(report, transcript_max_chars=10_000)
    assert payload["metadata"]["transcriptTruncated"] is False
    assert payload["metadata"]["transcriptOmittedChars"] == 0


def test_hubspot_engagement_no_network_or_env_access():
    with patch("aiohttp.ClientSession") as mock_session, patch.dict(os.environ, {}, clear=False):
        build_hubspot_engagement(_report())
        mock_session.assert_not_called()


def test_hubspot_engagement_does_not_mutate_input_report():
    report = _report(transcript_text="hello")
    original = copy.deepcopy(report.model_dump())
    payload = build_hubspot_engagement(report, contact_ids=["c1"])
    payload["associations"]["contactIds"].append("mutated")
    payload["customProperties"]["disposition"] = "mutated"
    assert report.model_dump() == original


def test_hubspot_engagement_output_json_serializable():
    payload = build_hubspot_engagement(_report(transcript_text="hi"), owner_id="o1")
    json.dumps(payload)  # must not raise


def test_hubspot_engagement_missing_optional_ids_default_sane():
    payload = build_hubspot_engagement(_report())
    assert payload["associations"]["ownerId"] is None
    assert payload["associations"]["contactIds"] == []


# --- Salesforce ---------------------------------------------------------------------------


def test_salesforce_task_shape_has_required_fields():
    payload = build_salesforce_task(_report(), who_id="who_1", what_id="what_1", owner_id="owner_1")
    assert payload["WhoId"] == "who_1"
    assert payload["WhatId"] == "what_1"
    assert payload["OwnerId"] == "owner_1"
    assert payload["Type"] == "Call"
    assert payload["CallDurationInSeconds"] == 42.0


def test_salesforce_task_transcript_truncated_with_metadata():
    report = _report(transcript_text="y" * 100)
    payload = build_salesforce_task(report, transcript_max_chars=10)
    assert payload["descriptionTruncated"] is True
    assert payload["descriptionOmittedChars"] > 0


def test_salesforce_task_does_not_mutate_input_report():
    report = _report(transcript_text="hello")
    original = copy.deepcopy(report.model_dump())
    payload = build_salesforce_task(report)
    payload["customFields"]["disposition"] = "mutated"
    assert report.model_dump() == original


def test_salesforce_task_no_network_or_sdk_access():
    with patch("aiohttp.ClientSession") as mock_session:
        build_salesforce_task(_report())
        mock_session.assert_not_called()


def test_salesforce_task_output_json_serializable():
    payload = build_salesforce_task(_report(transcript_text="hi"), who_id="w1")
    json.dumps(payload)


def test_salesforce_task_default_subject_when_not_provided():
    payload = build_salesforce_task(_report())
    assert payload["Subject"] == "Post-call report"


def test_salesforce_task_never_fabricates_ids():
    payload = build_salesforce_task(_report())
    assert payload["WhoId"] is None
    assert payload["WhatId"] is None
    assert payload["OwnerId"] is None
