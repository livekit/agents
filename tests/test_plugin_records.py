"""The records plugins build (decision 119): positional, mutable dataclasses."""

from __future__ import annotations

import dataclasses
import typing

import pytest

from livekit.agents import stt
from livekit.agents._generated import stt as generated_stt

pytestmark = pytest.mark.unit


# Catches: a plugin-facing record generated frozen, keyword-only or with annotations only the type
# checker can resolve: plugins build these by position, mutate them and read dataclasses.asdict.
def test_plugin_records_build_by_position_mutate_and_convert_to_dicts() -> None:
    data = stt.SpeechData("en_us", "hello", 0.5)
    event = stt.SpeechEvent(stt.SpeechEventType.FINAL_TRANSCRIPT, "req", [data])
    event.request_id = "changed"
    data.text = "hi"
    assert data.language == "en-US"
    dumped = dataclasses.asdict(event)
    assert dumped["request_id"] == "changed"
    assert dumped["alternatives"][0]["text"] == "hi"

    caps = stt.STTCapabilities(True, False)
    caps.diarization = True
    assert dataclasses.asdict(caps)["aligned_transcript"] is False
    # Every annotation is importable at runtime from the module that defines the records.
    for record in (stt.SpeechData, stt.SpeechEvent, stt.STTCapabilities):
        assert typing.get_type_hints(record, vars(generated_stt))
