"""The session report ships ``session.options`` as a log attribute. The OTel exporter
stringifies anything that is not a primitive, so an object left in the options (the turn
detector, for one) used to reach the cloud as ``<... object at 0x...>``. These tests pin the
descriptive form the serializer produces instead."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import AgentSession, inference
from livekit.agents.llm import ChatContext, ChatMessage, LLMError
from livekit.agents.metrics import LLMModelUsage, TTSModelUsage
from livekit.agents.telemetry.traces import (
    _describe_option_object,
    _serialize_option_value,
    _serialize_session_options,
)
from livekit.agents.tts import TTSError
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.events import CloseEvent, CloseReason, ErrorEvent
from livekit.agents.voice.report import SessionReport, _close_reason_name

pytestmark = pytest.mark.unit


def _turn_detection(serialized: dict[str, Any]) -> Any:
    return serialized["turn_handling"]["turn_detection"]


def _assert_report_safe(value: Any) -> None:
    """Everything left after serialization must be JSON primitives, lists, or dicts."""
    json.dumps(value)  # raises on anything the exporter would have to stringify
    if isinstance(value, dict):
        for v in value.values():
            _assert_report_safe(v)
    elif isinstance(value, list):
        for v in value:
            _assert_report_safe(v)
    else:
        assert value is None or isinstance(value, (str, bool, int, float))


def test_default_turn_detector_is_described_not_reprd() -> None:
    session = AgentSession()  # eager inference.TurnDetector() default
    serialized = _serialize_session_options(session.options)

    td = _turn_detection(serialized)
    assert isinstance(td, str)
    assert "object at 0x" not in td
    assert td.startswith("livekit.agents.inference.eot.detector.TurnDetector(")
    assert "model=turn-detector-" in td
    assert "provider=livekit" in td
    assert "sample_rate=16000" in td
    assert "local_fallback=True" in td
    # server-calibrated defaults in use: the override fields must be absent, not "NOT_GIVEN"
    assert "threshold_overrides" not in td
    assert "NOT_GIVEN" not in td
    _assert_report_safe(serialized)


def test_turn_detector_threshold_overrides_are_shown() -> None:
    td = inference.TurnDetector(
        version="v1-mini",
        unlikely_threshold=0.2,
        backchannel_threshold={"en": 0.7, "fr": 0.6},
    )
    desc = _describe_option_object(td)
    assert "threshold_overrides=0.2" in desc
    # dict overrides are rendered deterministically
    assert 'backchannel_threshold_overrides={"en": 0.7, "fr": 0.6}' in desc


def test_turn_detector_description_leaks_no_credentials() -> None:
    td = inference.TurnDetector(
        version="v1",
        base_url="https://inference.example.com",
        api_key="APIsecretkey123",
        api_secret="verysecretvalue",
    )
    desc = _describe_option_object(td)
    assert "APIsecretkey123" not in desc
    assert "verysecretvalue" not in desc
    assert "inference.example.com" not in desc


def test_mode_strings_pass_through() -> None:
    session = AgentSession(turn_handling={"turn_detection": "vad"})
    assert _turn_detection(_serialize_session_options(session.options)) == "vad"

    session = AgentSession(turn_handling={"turn_detection": "manual"})
    assert _turn_detection(_serialize_session_options(session.options)) == "manual"


def test_object_implementing_describe_options_is_rendered_with_them() -> None:
    class ThirdPartyDetector:
        def describe_options(self) -> dict[str, Any]:
            return {"model": "eou-v9", "provider": "acme", "thresholds": {"en": 0.7}}

    assert _describe_option_object(ThirdPartyDetector()) == (
        f'{__name__}.ThirdPartyDetector(model=eou-v9, provider=acme, thresholds={{"en": 0.7}})'
    )


def test_object_without_describe_options_is_its_class_name() -> None:
    # public attributes are not guessed at: a model-like object that does not opt in is
    # reported by class alone, however tempting its `model` looks
    class Opaque:
        model = "m"
        provider = "acme"

    assert _describe_option_object(Opaque()) == f"{__name__}.Opaque"


def test_describe_options_skips_none_and_not_given() -> None:
    class Sparse:
        def describe_options(self) -> dict[str, Any]:
            return {"model": "m", "provider": None, "label": NOT_GIVEN}

    assert _describe_option_object(Sparse()) == f"{__name__}.Sparse(model=m)"


def test_failing_describe_options_falls_back_to_the_class_name() -> None:
    class Broken:
        def describe_options(self) -> dict[str, Any]:
            raise RuntimeError("not ready")

    assert _describe_option_object(Broken()) == f"{__name__}.Broken"


def test_custom_sequence_and_set_values_keep_their_elements() -> None:
    # tts_text_transforms accepts any Sequence; a user-defined one must not collapse to
    # its class name
    class Transforms(Sequence[str]):
        def __init__(self, *items: str) -> None:
            self._items = items

        def __getitem__(self, i: Any) -> Any:
            return self._items[i]

        def __len__(self) -> int:
            return len(self._items)

    class Det:
        def describe_options(self) -> dict[str, Any]:
            return {"model": "m"}

    out = _serialize_option_value(
        {"tts_text_transforms": Transforms("filter_markdown", "filter_emoji"), "s": {2, 1}}
    )
    assert out == {
        "tts_text_transforms": ["filter_markdown", "filter_emoji"],
        "s": [1, 2],
    }
    assert _serialize_option_value(Transforms("a")) == ["a"]
    assert _serialize_option_value([Det()]) == [f"{__name__}.Det(model=m)"]
    _assert_report_safe(out)


def test_customer_prompt_text_is_omitted() -> None:
    # a keyterm-detection prompt override is customer-authored text; it never reaches the
    # report, at any nesting depth
    session = AgentSession(
        stt_context_options={
            "keyterms": ["Acme"],
            "keyterm_detection": {
                "enabled": True,
                "instructions": "Extract product names for Acme customer Jane Doe",
            },
        }
    )
    serialized = _serialize_session_options(session.options)
    detection = serialized["stt_context_options"]["keyterm_detection"]
    assert "instructions" not in detection
    assert detection["enabled"] is True
    assert serialized["stt_context_options"]["lk.pii.keyterms"] == ["Acme"]
    assert "Jane Doe" not in json.dumps(serialized)

    assert _serialize_option_value({"a": {"instructions": "x", "keep": 1}}) == {"a": {"keep": 1}}


def test_nested_containers_and_key_aliases() -> None:
    class Det:
        def describe_options(self) -> dict[str, Any]:
            return {"model": "m"}

    value = {
        "keyterms": ["LiveKit", "Acme"],
        "nested": {"detector": Det(), "flags": (True, 1, 2.5, None)},
    }
    out = _serialize_option_value(value)
    assert out == {
        "lk.pii.keyterms": ["LiveKit", "Acme"],
        "nested": {"detector": f"{__name__}.Det(model=m)", "flags": [True, 1, 2.5, None]},
    }
    _assert_report_safe(out)


def _report(**kwargs: Any) -> SessionReport:
    return SessionReport(
        job_id="job-1",
        room_id="room-1",
        room="test-room",
        options=AgentSession().options,
        events=kwargs.pop("events", []),
        chat_history=ChatContext(kwargs.pop("chat_items", [])),
        **kwargs,
    )


def test_report_to_dict_default_keys() -> None:
    # the keys before agent_name keep their order; the rest are appended
    assert list(_report().to_dict()) == [
        "job_id",
        "room_id",
        "room",
        "events",
        "audio_recording_path",
        "audio_recording_started_at",
        "options",
        "chat_history",
        "timestamp",
        "usage",
        "sdk_version",
        "agent_name",
        "tags",
        "duration_ms",
        "user_turns",
        "errors",
        "close_reason",
    ]


def test_report_to_dict_flags_omit_keys() -> None:
    report = _report()
    full = report.to_dict()

    assert report.to_dict(include_chat_history=False) == {
        k: v for k, v in full.items() if k != "chat_history"
    }
    assert report.to_dict(include_events=False) == {k: v for k, v in full.items() if k != "events"}


def _llm_error() -> LLMError:
    return LLMError(timestamp=0.0, label="llm", error=RuntimeError("x"), recoverable=True)


def _tts_error() -> TTSError:
    return TTSError(timestamp=0.0, label="tts", error=RuntimeError("x"), recoverable=True)


def test_report_to_dict_computed_keys() -> None:
    report = _report(
        agent_name="support",
        tags=["vip"],
        duration=12.5,
        chat_items=[
            ChatMessage(role="user", content=["hi"]),
            ChatMessage(role="assistant", content=["hello"]),
            ChatMessage(role="user", content=["bye"]),
        ],
        model_usage=[
            LLMModelUsage(
                provider="openai",
                model="gpt-4o",
                input_tokens=10,
                output_tokens=5,
                input_cached_tokens=2,
            ),
            TTSModelUsage(provider="cartesia", model="sonic", characters_count=40),
        ],
        events=[
            ErrorEvent(error=_llm_error(), source=MagicMock()),
            ErrorEvent(error=_llm_error(), source=MagicMock()),
            ErrorEvent(error=_tts_error(), source=MagicMock()),
            CloseEvent(reason=CloseReason.ERROR),
            CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED),
        ],
    )

    out = report.to_dict()

    assert out["usage"] == [
        {
            "type": "llm_usage",
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 10,
            "output_tokens": 5,
            "input_cached_tokens": 2,
        },
        {"type": "tts_usage", "provider": "cartesia", "model": "sonic", "characters_count": 40},
    ]
    assert out["agent_name"] == "support"
    assert out["tags"] == ["vip"]
    assert out["duration_ms"] == 12500
    assert out["user_turns"] == 2
    assert out["errors"] == {"llm": 2, "stt": 0, "tts": 1, "realtime": 0, "interruption": 0}
    # the last close event wins
    assert out["close_reason"] == "CR_PARTICIPANT_DISCONNECTED"


def test_report_to_dict_computed_keys_empty_session() -> None:
    out = _report().to_dict()

    assert out["usage"] is None
    assert out["agent_name"] == ""
    assert out["tags"] == []
    assert out["duration_ms"] is None
    assert out["user_turns"] == 0
    assert out["errors"] == {"llm": 0, "stt": 0, "tts": 0, "realtime": 0, "interruption": 0}
    assert out["close_reason"] is None


def test_report_close_reasons_match_proto_enum() -> None:
    # a new SDK close reason the proto enum lacks would fall back to CR_UNKNOWN; fail here instead
    for reason in CloseReason:
        assert _close_reason_name(reason) != "CR_UNKNOWN", reason
    assert _close_reason_name(CloseReason.TASK_COMPLETED) == "CR_COMPLETED"
