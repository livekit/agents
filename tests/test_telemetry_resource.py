"""Unit tests for ``livekit.agents.telemetry.traces._resource``: ``Resource.create`` with the
resource detectors run once per process."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

from livekit.agents.telemetry import traces

pytestmark = pytest.mark.unit


@pytest.fixture
def detections(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    monkeypatch.setattr(traces, "_detected_resource", None)
    calls: list[int] = []
    create = Resource.create

    def _counting_create(*args, **kwargs) -> Resource:
        calls.append(1)
        return create(*args, **kwargs)

    monkeypatch.setattr(Resource, "create", staticmethod(_counting_create))
    return calls


def test_matches_resource_create_and_detects_once(detections: list[int]) -> None:
    attrs = {SERVICE_NAME: "livekit-agents", "room_id": "RM_x"}
    first = traces._resource(attrs)
    second = traces._resource({})
    assert len(detections) == 1

    detections.clear()
    assert first.attributes == Resource.create(attrs).attributes
    assert second.attributes == Resource.create({}).attributes


def test_detects_again_after_fork(detections: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    traces._resource({})
    monkeypatch.setattr(traces.os, "getpid", lambda: -1)
    traces._resource({})
    assert len(detections) == 2
