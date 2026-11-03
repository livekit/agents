"""The session report's host records: the tagger hands back report tags and outcomes."""

import pytest

from livekit.agents import _core
from livekit.agents._generated.telemetry import ReportTag

pytestmark = pytest.mark.unit


def test_tagger_entries_are_the_reports_tags() -> None:
    tagger = _core.Tagger()
    tagger.add("slot:booked", metadata={"slot": "abc"})
    tagger.fail(reason="hung up")
    entries = tagger.entries()
    assert all(isinstance(entry, ReportTag) for entry in entries)
    assert [entry.name for entry in entries] == ["slot:booked", "lk.fail"]
    assert entries[0].metadata == {"slot": "abc"} and entries[1].metadata is None
    assert entries[0].timestamp <= entries[1].timestamp
    assert tagger.outcome == "fail" and tagger.outcome_reason == "hung up"
