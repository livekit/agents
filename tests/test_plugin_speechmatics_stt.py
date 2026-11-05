"""Offline Speechmatics configuration, message translation, and stream lifecycle tests.

Run with ``uv run pytest tests/test_plugin_speechmatics_stt.py --plugin speechmatics``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError
from speechmatics.agent_stt import (
    DEFAULT_MODEL,
    AudioEncoding,
    ServerMessageType,
    TurnConfig,
    TurnDetectionMode as AgentTurnDetectionMode,
)

from livekit.agents import APIConnectOptions, APIError, stt
from livekit.agents.utils.aio import ChanEmpty
from livekit.plugins.speechmatics import stt as speechmatics_stt
from livekit.plugins.speechmatics.stt import (
    DEFAULT_BASE_URL,
    SpeechStream,
    TurnDetectionMode,
)

pytestmark = pytest.mark.plugin("speechmatics")

# `transcription_config` is `additionalProperties: false` in the agent input spec. Anything
# outside this list is rejected by the service, which is what made every pre-migration
# request fail. Kept as a literal so an SDK that starts emitting a new field is caught here
# rather than on the wire.
WIRE_WHITELIST = frozenset(
    {
        "additional_vocab",
        "diarization",
        "domain",
        "emit_sentences",
        "enable_partials",
        "language",
        "model",
        "operating_point",
        "output_locale",
        "speaker_diarization_config",
        "transcript_filtering_config",
    }
)

# The pre-Agent-STT operating points, by what a user would have written.
LEGACY_OPERATING_POINTS = ("enhanced", "standard")


def _stt(**kwargs: Any) -> speechmatics_stt.STT:
    """An STT with credentials, so a test fails on what it is about and not on auth."""
    return speechmatics_stt.STT(api_key="test-key", **kwargs)


def _stream(**kwargs: Any) -> SpeechStream:
    """A stream whose connect task is cancelled before it can run.

    `RecognizeStream.__init__` schedules `_main_task`, which would dial the service. A
    scheduled task cannot start until the test awaits something, so cancelling it here
    leaves the message handlers callable with no socket in sight.
    """
    stream = _stt(**kwargs).stream()
    stream._task.cancel()
    return stream


def _events(stream: SpeechStream) -> list[stt.SpeechEvent]:
    """Everything the stream has emitted so far."""
    drained: list[stt.SpeechEvent] = []
    while True:
        try:
            drained.append(stream._event_ch.recv_nowait())
        except ChanEmpty:
            return drained


def _segment_message(
    *,
    final: bool = True,
    transcript: str = "hello world",
    speaker: str | None = "S1",
    start: float = 1.5,
    end: float = 2.5,
) -> dict[str, Any]:
    """An `AddSegment` / `AddPartialSegment` as agent-STT sends it.

    Note the shape: a **singular** `segment`, and timings on the **message**. The voice SDK
    expected a `segments` list with per-segment metadata, which is why the plugin emitted
    no transcripts at all before the migration.
    """
    message_type = ServerMessageType.ADD_SEGMENT if final else ServerMessageType.ADD_PARTIAL_SEGMENT
    return {
        "message": message_type.value,
        "segment": {"transcript": transcript, "speaker": speaker},
        "metadata": {"start_time": start, "end_time": end},
    }


class _StubClient:
    """Stands in for `AgentSttAsyncClient` where only `finalize()` is under test."""

    def __init__(self) -> None:
        self.is_connected = True
        self.finalize_calls = 0

    def finalize(self) -> None:
        self.finalize_calls += 1


# --- Construction and validation -----------------------------------------------------


def test_capabilities_mirror_resolved_options() -> None:
    """Capabilities must be built from the resolved options, not the raw arguments."""
    assert _stt().capabilities.interim_results is True
    assert _stt(include_partials=False).capabilities.interim_results is False

    assert _stt(enable_diarization=False).capabilities.diarization is False


def test_audio_format_is_validated() -> None:
    """Both values are fixed, and a wrong one must fail here rather than mid-session.

    The encoding check is deliberately narrower than the spec: `pcm_f32le` is legal on the
    wire, but LiveKit frames are int16 and we forward them unconverted, so declaring it
    would have the service read 2-byte samples as 4-byte floats — plausible garbage, no
    error.
    """
    with pytest.raises(ValueError, match="sample_rate"):
        _stt(sample_rate=8000)

    with pytest.raises(ValueError, match="audio_encoding"):
        _stt(audio_encoding=AudioEncoding.PCM_F32LE)


# --- Model resolution ----------------------------------------------------------------


@pytest.mark.parametrize("value", ("lindne-1", *LEGACY_OPERATING_POINTS))
def test_unsupported_model_falls_back(value: str, caplog) -> None:
    """Nothing a caller can name breaks construction — a typo and the old RT operating
    points alike land on the default model.

    `enhanced` is the one that matters: the pre-migration README's own examples used it.
    """
    with caplog.at_level("WARNING"):
        instance = _stt(model=value)

    assert instance.model == DEFAULT_MODEL.value
    assert value in caplog.text


# --- Turn detection ------------------------------------------------------------------


def test_turn_config_is_a_top_level_sibling() -> None:
    """Turn detection is not a `transcription_config` field, and never was on this SDK.

    Sending it as one raised `TypeError` at config construction, which made every
    `STT.stream()` fail unconditionally. It travels as a top-level `turn_config`, and must
    carry the SDK's own enum member because `TurnConfig.to_dict()` reads the member value.
    """
    assert "turn_detection_mode" not in _stt()._prepare_config().to_dict()

    for mode, expected in (
        (TurnDetectionMode.VAD, AgentTurnDetectionMode.VAD),
        (TurnDetectionMode.EXTERNAL, AgentTurnDetectionMode.EXTERNAL),
    ):
        member = speechmatics_stt._handle_turn_detection_mode(mode)
        assert member is expected
        assert TurnConfig(turn_detection_mode=member).to_dict() == {
            "turn_detection_mode": mode.value
        }


# --- VAD and finalize ----------------------------------------------------------------


def test_bare_external_loads_a_vad() -> None:
    """`EXTERNAL` is the default and the service does not endpoint in it.

    The pre-Agent-STT plugin auto-loaded Silero here. Without it a bare `STT()` finalizes
    nothing and transcribes nothing, which is the zero-config path.
    """
    pytest.importorskip("livekit.plugins.silero")

    assert _stt()._vad is not None


def test_a_missing_silero_still_constructs(monkeypatch, caplog) -> None:
    """The auto-load is best-effort: the old hard `ImportError` made the plugin unusable
    wherever the optional package was absent."""
    monkeypatch.setitem(sys.modules, "livekit.plugins.silero", None)

    with caplog.at_level("WARNING"):
        instance = _stt()

    assert instance._vad is None
    assert "EXTERNAL turn-detection mode with no `vad`" in caplog.text


def test_explicit_none_vad_opts_out(caplog) -> None:
    """`vad=None` is the advanced path: that caller drives `finalize()` by hand."""
    with caplog.at_level("WARNING"):
        instance = _stt(vad=None)

    assert instance._vad is None
    assert "EXTERNAL turn-detection mode with no `vad`" in caplog.text


async def test_finalize_and_vad_only_act_in_external() -> None:
    """In `VAD` mode the service closes turns, so neither may fire.

    Both halves are the same guard against double endpointing: the plugin asking for a turn
    boundary while the service is already deciding them.
    """
    external = _stream(vad=None)
    external._client = _StubClient()

    external._stt.finalize()
    assert external._client.finalize_calls == 1

    service_side = _stream(turn_detection_mode=TurnDetectionMode.VAD)
    service_side._client = _StubClient()

    assert service_side._stt._vad is None  # nothing is auto-loaded outside EXTERNAL
    service_side._stt.finalize()
    assert service_side._client.finalize_calls == 0


# --- Config down-translation ---------------------------------------------------------


def test_config_sends_only_whitelisted_fields() -> None:
    """The wire contract. `transcription_config` is `additionalProperties: false`, so one
    stray field rejects the whole session — which is exactly what used to happen.

    Also covers the per-stream `language` override, which was being ignored in favour of
    the constructor value.
    """
    config = _stt(
        language="en",
        output_locale="en-US",
        domain="finance",
        include_partials=True,
        max_speakers=4,
    )._prepare_config()
    sent = config.to_dict()

    assert set(sent) <= WIRE_WHITELIST, f"not on the wire whitelist: {set(sent) - WIRE_WHITELIST}"
    assert sent["language"] == "en"
    assert sent["model"] == DEFAULT_MODEL.value
    assert sent["output_locale"] == "en-US"
    assert sent["domain"] == "finance"
    assert sent["enable_partials"] is True
    assert json.dumps(sent)

    assert _stt(language="en")._prepare_config("fr").to_dict()["language"] == "fr"


def test_diarization_config_built_only_when_used() -> None:
    """These four knobs were accepted but never sent before the sub-config was wired.

    An empty config is never sent either: `None` means the key is absent from the wire.
    """
    assert _stt()._prepare_config().to_dict().get("speaker_diarization_config") is None
    assert _stt(enable_diarization=False, max_speakers=4)._prepare_config().diarization is None

    config = _stt(
        max_speakers=4,
        speaker_sensitivity=0.7,
        prefer_current_speaker=True,
        known_speakers=[{"label": "S1", "speaker_identifiers": ["x"]}],
    )._prepare_config()
    diarization = config.to_dict()["speaker_diarization_config"]

    assert config.to_dict()["diarization"] == "speaker"
    assert diarization["max_speakers"] == 4
    assert diarization["speaker_sensitivity"] == 0.7
    assert diarization["prefer_current_speaker"] is True
    assert diarization["speakers"] == [{"label": "S1", "speaker_identifiers": ["x"]}]


def test_additional_vocab_accepts_both_classes() -> None:
    """Either SDK's entry is fine, and the assertion runs to the JSON encode.

    That is where the wrong one actually failed: the config carries an unrecognised object
    into `additional_vocab` untouched, so an old pydantic entry survived construction and
    then killed the session at `StartRecognition`. The plugin duck-types entries on their
    fields, so a stand-in with `speechmatics.voice`'s two fields represents that SDK, which
    the plugin does not depend on.
    """
    from speechmatics.agent_stt import AdditionalVocabEntry as AgentEntry

    for entry_type in (AgentEntry, SimpleNamespace):
        entry = entry_type(content="Speechmatics", sounds_like=["speech matics"])
        sent = _stt(additional_vocab=[entry])._prepare_config().to_dict()

        assert json.dumps(sent)
        assert sent["additional_vocab"] == [
            {"content": "Speechmatics", "sounds_like": ["speech matics"]}
        ]

    # The residue: the exported class is the dataclass, so pydantic's API is gone from it.
    assert not hasattr(AgentEntry(content="x"), "model_dump")


def test_unreadable_vocab_entry_fails_at_construction() -> None:
    """Better than the `TypeError` the JSON encode would raise once the session opens."""
    with pytest.raises(ValueError, match="no `content`"):
        _stt(additional_vocab=[object()])


# --- Message up-translation ----------------------------------------------------------


async def test_add_segment_becomes_final_transcript() -> None:
    """The singular-segment seam, and the full `SpeechData` shape LiveKit consumes.

    `language` has no wire field and is filled from the session config; timings come from
    the message-level `metadata` and carry `start_time_offset`, which is what keeps them
    monotonic across a mid-stream reconnect.
    """
    stream = _stream(language="en")
    stream.start_time_offset = 10.0

    stream._handle_message(_segment_message(transcript="hello world", start=1.5, end=2.5))
    events = _events(stream)

    assert [event.type for event in events] == [stt.SpeechEventType.FINAL_TRANSCRIPT]
    data = events[0].alternatives[0]
    assert data.text == "hello world"
    assert data.speaker_id == "S1"
    assert data.language == "en"
    assert data.start_time == 11.5
    assert data.end_time == 12.5


async def test_add_partial_segment_becomes_interim() -> None:
    """Same payload seam, and the message type is the only thing that selects the event."""
    stream = _stream()

    stream._handle_message(_segment_message(final=False, transcript="hel"))
    events = _events(stream)

    assert [event.type for event in events] == [stt.SpeechEventType.INTERIM_TRANSCRIPT]
    assert events[0].alternatives[0].text == "hel"


async def test_turn_messages_become_speech_events() -> None:
    """`StartOfTurn`/`EndOfTurn` are turn signals with no payload we read.

    The usage event is synthesised on `EndOfTurn`, and only when audio actually flowed —
    otherwise a turn with no audio would bill zero and emit noise.
    """
    stream = _stream()

    stream._handle_message({"message": ServerMessageType.START_OF_TURN.value})
    stream._handle_message({"message": ServerMessageType.END_OF_TURN.value})

    assert [event.type for event in _events(stream)] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.END_OF_SPEECH,
    ]

    stream._speech_duration = 3.0
    stream._handle_message({"message": ServerMessageType.END_OF_TURN.value})
    events = _events(stream)

    assert [event.type for event in events] == [
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.RECOGNITION_USAGE,
    ]
    assert events[1].recognition_usage is not None
    assert events[1].recognition_usage.audio_duration == 3.0
    assert stream._speech_duration == 0  # reset, so the next turn does not double-bill


async def test_speaker_id_formatting_and_uu_fallback() -> None:
    """`speaker_format` is the last live remnant of the old formatter.

    `UU` covers segments the service did not attribute — which is every segment when
    diarization is off, so a format string renders `[Speaker UU]` for all of them.
    """
    stream = _stream(speaker_format="@{speaker_id}: {text}")
    stream._handle_message(_segment_message(transcript="hi", speaker="S2"))

    assert _events(stream)[0].alternatives[0].text == "@S2: hi"

    bare = _stream()
    bare._handle_message(_segment_message(transcript="hi", speaker=None))
    data = _events(bare)[0].alternatives[0]

    assert data.speaker_id == "UU"
    assert data.text == "hi"  # no format string, so the label is not rendered into the text


async def test_server_error_raises_and_warning_only_logs(caplog) -> None:
    """An `Error` ends the session. It used to be logged and then the stream hung until
    timeout with the reason lost; now it surfaces as a non-retryable `APIError`."""
    stream = _stream()
    message = {"message": ServerMessageType.ERROR.value, "type": "invalid_config", "reason": "nope"}

    with pytest.raises(APIError, match="invalid_config") as raised:
        stream._handle_message(message)

    assert raised.value.retryable is False
    assert raised.value.body == message

    # A Warning does not end the session, so it must not raise.
    with caplog.at_level("WARNING"):
        stream._handle_message({"message": ServerMessageType.WARNING.value, "reason": "slow"})

    assert "Warning" in caplog.text
    assert _events(stream) == []


async def test_speakers_result_is_stored_not_emitted() -> None:
    """It answers `get_speaker_ids()` rather than becoming a `SpeechEvent`."""
    stream = _stream()
    speakers = [{"label": "S1", "speaker_identifiers": ["x"]}]

    stream._handle_message(
        {"message": ServerMessageType.SPEAKERS_RESULT.value, "speakers": speakers}
    )

    assert stream._speaker_result == [
        speechmatics_stt.SpeakerIdentifier(**item) for item in speakers
    ]
    assert stream._speaker_result_event.is_set()
    assert _events(stream) == []


# --- Removed arguments ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ("operating_point", "enable_partials", "focus_speakers", "end_of_utterance_mode")
)
def test_removed_arguments_are_rejected(name: str) -> None:
    """The pre-Agent-STT arguments fail at construction instead of being ignored."""
    with pytest.raises(TypeError):
        _stt(vad=None, **{name: "whatever"})

    assert not hasattr(speechmatics_stt.STT, "update_speakers")


def test_non_agent_endpoint_is_reported(monkeypatch, caplog) -> None:
    """The old `/v2` default, and any `SPEECHMATICS_RT_URL` still pointing at it.

    The env var survives an upgrade untouched, so this is the invisible case: the session
    connects, speaks a protocol the plugin cannot read, and returns nothing.
    """
    monkeypatch.setenv("SPEECHMATICS_RT_URL", "wss://eu2.rt.speechmatics.com/v2")

    with caplog.at_level("WARNING"):
        _stt(vad=None)

    assert "/v2/agent" in caplog.text
    assert "no transcripts" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        _stt(vad=None, base_url=f"{DEFAULT_BASE_URL}/")  # a trailing slash is still an endpoint

    assert "not an Agent STT endpoint" not in caplog.text


async def test_invalid_speaker_reply_fails_request_without_stopping_recognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SpeakerClient:
        def __init__(self) -> None:
            self.is_connected = False
            self.connected = asyncio.Event()
            self.handlers: dict[ServerMessageType, Callable[[dict[str, object]], None]] = {}
            self.speakers: list[dict[str, object]] = [{"label": "Agent", "speaker_identifiers": 7}]

        def on(
            self, event: ServerMessageType, callback: Callable[[dict[str, object]], None]
        ) -> None:
            self.handlers[event] = callback

        async def __aenter__(self) -> SpeakerClient:
            self.is_connected = True
            self.connected.set()
            return self

        async def __aexit__(self, *args: object) -> None:
            await self.disconnect()

        async def send_message(self, message: dict[str, object]) -> None:
            self.handlers[ServerMessageType.SPEAKERS_RESULT](
                {"message": ServerMessageType.SPEAKERS_RESULT.value, "speakers": self.speakers}
            )

        async def disconnect(self) -> None:
            self.is_connected = False

    client = SpeakerClient()
    monkeypatch.setattr(speechmatics_stt, "AgentSttAsyncClient", lambda **kwargs: client)
    provider = _stt(vad=None)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await asyncio.wait_for(client.connected.wait(), timeout=0.5)
        with pytest.raises(APIError, match="invalid speaker identifiers") as error:
            await asyncio.wait_for(provider.get_speaker_ids(), timeout=0.5)
        assert isinstance(error.value.__cause__, ValidationError)
        assert not stream._task.done()

        client.speakers = [{"label": "Agent", "speaker_identifiers": ["voice-1"]}]
        speakers = await asyncio.wait_for(provider.get_speaker_ids(), timeout=0.5)
        assert len(speakers) == 1
        assert speakers[0].label == "Agent"
        assert speakers[0].speaker_identifiers == ["voice-1"]

        client.handlers[ServerMessageType.ADD_SEGMENT](
            _segment_message(transcript="Still listening")
        )
        event = await asyncio.wait_for(anext(stream), timeout=0.5)
        assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        assert event.alternatives[0].text == "Still listening"
        assert not stream._task.done()
    finally:
        await stream.aclose()
