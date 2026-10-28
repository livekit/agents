"""Cloud refusal stops SDK requests without inventing a successful server response."""

import pytest
import requests

from livekit.agents import _core
from livekit.agents.telemetry import _destination

pytestmark = pytest.mark.unit


def test_recording_disabled_stops_requests_and_preserves_failure(monkeypatch):
    calls = []
    response = requests.Response()
    response.status_code = 401
    response._content = b"data recording is disabled for project"

    def request(self, *args, **kwargs):
        calls.append(args)
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    # Keep this test independent of another job's process-wide gate.
    monkeypatch.setattr(_destination, "upload_gate", _core.UploadGate(), raising=False)
    session = _destination._AuthRefreshingSession(lambda: {"Authorization": "Bearer fresh"})
    with pytest.raises(_destination._RecordingDisabled, match="recording is disabled"):
        session.get("https://cloud.invalid/observability")
    with pytest.raises(_destination._RecordingDisabled, match="recording is disabled"):
        session.get("https://cloud.invalid/observability")
    assert len(calls) == 1
    assert response.status_code == 401
