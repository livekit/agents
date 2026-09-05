"""The session report ships ``session.options`` as a log attribute. The OTel exporter
stringifies anything that is not a primitive, so an object left in the options (the turn
detector, for one) used to reach the cloud as ``<... object at 0x...>``. These tests pin the
descriptive form the serializer produces instead."""

from __future__ import annotations

import json
from typing import Any

import pytest

from livekit.agents import AgentSession, inference
from livekit.agents.telemetry.traces import (
    _describe_option_object,
    _serialize_option_value,
    _serialize_session_options,
)
from livekit.agents.types import NOT_GIVEN

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
    assert td.startswith("TurnDetector(")
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


def test_plugin_like_detector_uses_model_and_provider() -> None:
    class ThirdPartyDetector:
        @property
        def model(self) -> str:
            return "eou-v9"

        @property
        def provider(self) -> str:
            return "acme"

        # a method that happens to share a whitelisted name must never be rendered
        def label(self) -> str:
            return "not-an-attribute"

        async def unlikely_threshold(self, language: str | None) -> float | None:
            return 0.5

    assert _describe_option_object(ThirdPartyDetector()) == (
        "ThirdPartyDetector(model=eou-v9, provider=acme)"
    )


def test_object_without_descriptors_falls_back_to_class_name() -> None:
    class Opaque:
        pass

    assert _describe_option_object(Opaque()) == "Opaque()"


def test_not_given_and_none_attributes_are_skipped() -> None:
    class Sparse:
        model = "m"
        provider = None
        label = NOT_GIVEN

    assert _describe_option_object(Sparse()) == "Sparse(model=m)"


def test_nested_containers_and_key_aliases() -> None:
    class Det:
        model = "m"

    value = {
        "keyterms": ["LiveKit", "Acme"],
        "nested": {"detector": Det(), "flags": (True, 1, 2.5, None)},
    }
    out = _serialize_option_value(value)
    assert out == {
        "lk.pii.keyterms": ["LiveKit", "Acme"],
        "nested": {"detector": "Det(model=m)", "flags": [True, 1, 2.5, None]},
    }
    _assert_report_safe(out)
