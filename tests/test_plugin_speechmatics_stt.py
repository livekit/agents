"""Offline tests for the Speechmatics Agent STT plugin.

Two halves, and they answer different questions:

- **The plugin's own behaviour** — what a caller configures, what reaches the wire, and
  what the service's messages become. Every test covers one translation across one seam,
  so a failure names the thing that broke rather than "Speechmatics is broken".
- **Back-compatibility** — the public surface the plugin exposed before Agent STT. Those
  names and arguments still resolve, and each one says so in the log. Deleted after
  2026-10-05, along with the `speechmatics-voice` dependency they are served from.

Nothing here opens a socket. Live coverage is the k8s-saas e2e suite, which drives the
real plugin against `/v2/agent`; duplicating it here would only be slower and flakier.

Run standalone:

    .venv/bin/python -m pytest tests/test_plugin_speechmatics_stt.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest
from speechmatics.agent_stt import (
    DEFAULT_MODEL,
    AudioEncoding,
    Model,
    ServerMessageType,
    TurnConfig,
    TurnDetectionMode as AgentTurnDetectionMode,
)

from livekit.agents import APIError, stt
from livekit.agents.utils.aio.channel import ChanEmpty
from livekit.plugins.speechmatics import stt as speechmatics_stt
from livekit.plugins.speechmatics.stt import (
    DEFAULT_BASE_URL,
    DEFAULT_TURN_DETECTION_MODE,
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

# The pre-Agent-STT public names, by what a user would have written.
OPERATING_POINT = "OperatingPoint"
SPEAKER_FOCUS_MODE = "SpeakerFocusMode"
REMOVED_TURN_MODES = ("ADAPTIVE", "FIXED", "SMART_TURN")
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
    stream._metrics_task.cancel()
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


@pytest.fixture(autouse=True)
def _reset_deprecation_warnings():
    """Each symbol shim warns once per process, so tests would otherwise shadow each other."""
    import livekit.plugins.speechmatics as plugin

    plugin._warned_deprecated.clear()
    yield
    plugin._warned_deprecated.clear()


# --- Construction and validation -----------------------------------------------------


def test_defaults() -> None:
    """The zero-config contract, in one place, because it is what everyone gets."""
    instance = _stt()
    opts = instance._stt_options

    assert opts.model == DEFAULT_MODEL.value == "linden-1"
    assert opts.turn_detection_mode is DEFAULT_TURN_DETECTION_MODE is TurnDetectionMode.EXTERNAL
    assert instance._base_url == DEFAULT_BASE_URL
    assert instance._base_url.endswith("/v2/agent")
    # Diarization was on by default pre-migration (every voice preset set it), and the
    # `None` sentinel that used to mean "keep the preset" now reads as falsy.
    assert opts.enable_diarization is True
    assert instance._sample_rate == 16000
    assert instance._audio_encoding is AudioEncoding.PCM_S16LE


def test_capabilities_mirror_resolved_options() -> None:
    """Capabilities must be built from the resolved options, not the raw arguments.

    `enable_partials` arrives deprecated and is migrated to `include_partials`. Computing
    capabilities first advertised `interim_results=True` while the service was told to
    withhold partials.
    """
    assert _stt().capabilities.interim_results is True

    for kwargs in ({"include_partials": False}, {"enable_partials": False}):
        assert _stt(**kwargs).capabilities.interim_results is False

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


def test_model_defaults_and_accepts_supported() -> None:
    """`linden-1` is the default and, today, the only model. Enum or string, same result."""
    assert _stt().model == DEFAULT_MODEL.value

    for value in (Model.LINDEN_1, "linden-1"):
        assert _stt(model=value).model == "linden-1"


def test_operating_point_alias_and_conflict(caplog) -> None:
    """The deprecated alias still works, and `model` wins when both are given.

    The warning goes through `logger`, not `warnings.warn`: a `DeprecationWarning` is
    silenced outside `__main__`, so this deprecation printed nothing in a real agent.
    """
    with caplog.at_level("WARNING"):
        assert _stt(operating_point="linden-1").model == "linden-1"
    assert "`operating_point` is deprecated" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        assert _stt(model="linden-1", operating_point="enhanced").model == "linden-1"
    assert "using 'linden-1'" in caplog.text


@pytest.mark.parametrize("value", ("lindne-1", *LEGACY_OPERATING_POINTS))
def test_unsupported_model_falls_back(value: str, caplog) -> None:
    """Nothing a caller can name breaks construction — a typo and the old RT operating
    points alike land on the default model.

    `enhanced` is the one that matters: the pre-migration README's own examples used it.
    """
    with caplog.at_level("WARNING"):
        instance = _stt(operating_point=value)

    assert instance.model == DEFAULT_MODEL.value
    assert value in caplog.text


# --- Turn detection ------------------------------------------------------------------


def test_default_mode_is_external() -> None:
    """Parity with the pre-Agent-STT default. Pinned so a future flip is deliberate."""
    assert DEFAULT_TURN_DETECTION_MODE is TurnDetectionMode.EXTERNAL
    assert _stt().capabilities.streaming is True
    assert _stt()._stt_options.turn_detection_mode is TurnDetectionMode.EXTERNAL


@pytest.mark.parametrize("member", REMOVED_TURN_MODES)
def test_deprecated_modes_resolve_to_the_default(member: str, caplog) -> None:
    """All three named a service-side strategy agent-STT does not distinguish.

    Asserted against the constant, not a literal, so moving the default moves the test
    with it instead of turning it red.
    """
    mode = getattr(TurnDetectionMode, member)
    assert mode.value == member.lower()  # the values the old plugin accepted

    with caplog.at_level("WARNING"):
        resolved = speechmatics_stt._resolve_turn_detection_mode(mode)

    assert resolved is DEFAULT_TURN_DETECTION_MODE
    assert member in caplog.text
    assert "deprecated" in caplog.text


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
    then killed the session at `StartRecognition`.
    """
    from speechmatics.agent_stt import AdditionalVocabEntry as AgentEntry
    from speechmatics.voice import AdditionalVocabEntry as VoiceEntry

    for entry_type in (AgentEntry, VoiceEntry):
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

    assert stream._speaker_result == speakers
    assert stream._speaker_result_event.is_set()
    assert _events(stream) == []


# --- Back-compatibility: the pre-Agent-STT public surface ----------------------------


def test_operating_point_still_imports_and_warns(caplog) -> None:
    """`from ... import OperatingPoint` used to fail on upgrade, before any `STT` existed.

    Served from the voice SDK that declared it, so callers get the identical class object
    they already hold and their `isinstance` checks still pass.
    """
    import speechmatics.rt
    import speechmatics.voice

    with caplog.at_level("WARNING"):
        from livekit.plugins.speechmatics import OperatingPoint

    assert OperatingPoint is speechmatics.voice.OperatingPoint
    # voice only re-exported rt's, so the two are one class and either import site matches.
    assert OperatingPoint is speechmatics.rt.OperatingPoint
    assert {point.value for point in OperatingPoint} == set(LEGACY_OPERATING_POINTS)
    assert OPERATING_POINT in caplog.text
    assert "deprecated" in caplog.text


def test_speaker_focus_mode_still_imports_and_warns(caplog) -> None:
    """Speaker focus has no Agent STT equivalent, so the name resolves and says so."""
    import speechmatics.voice

    with caplog.at_level("WARNING"):
        from livekit.plugins.speechmatics import SpeakerFocusMode

    assert SpeakerFocusMode is speechmatics.voice.SpeakerFocusMode
    assert {member.name for member in SpeakerFocusMode} == {"RETAIN", "IGNORE"}
    assert SPEAKER_FOCUS_MODE in caplog.text
    assert "not supported by Agent STT" in caplog.text


def test_unknown_attributes_still_raise() -> None:
    """The shim is a lookup of known names, not a catch-all that invents attributes."""
    import livekit.plugins.speechmatics as plugin

    with pytest.raises(AttributeError, match="NotAThing"):
        _ = plugin.NotAThing


def test_voice_sdk_stays_off_the_runtime_path() -> None:
    """What keeps "this PR removes the voice SDK" true.

    The shim imports live inside the `__getattr__` branches, so importing the plugin,
    constructing `STT`, building the config and passing deprecated kwargs all leave
    `speechmatics.voice` out of `sys.modules`. Only touching a deprecated *symbol* loads it.

    Runs in a subprocess because import state is process-wide: any earlier test in this
    file has already pulled voice in.
    """
    probe = """
import sys
from livekit.plugins.speechmatics import STT

stt = STT(api_key="k", vad=None, focus_mode="retain", operating_point="enhanced")
stt._prepare_config()
assert "speechmatics.voice" not in sys.modules, "voice SDK reached the runtime path"
print("clean")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_update_speakers_is_a_warning_no_op(caplog) -> None:
    """The method only ever updated speaker focus, so there is nothing left for it to do.

    It stays callable so old code keeps running instead of dying on `AttributeError`.
    """
    instance = _stt(vad=None)

    with caplog.at_level("WARNING"):
        assert instance.update_speakers(focus_speakers=["S1"], ignore_speakers=["S2"]) is None

    assert "update_speakers" in caplog.text
    assert "not supported by Agent STT" in caplog.text


@pytest.mark.parametrize("name", sorted(speechmatics_stt._DROPPED_ARGS))
def test_dropped_kwargs_are_reported(name: str, caplog) -> None:
    """Accepted so upgrading does not break construction, but they reach nothing.

    Each warning carries the reason, because "deprecated and no longer used" does not tell
    a caller whether to go looking for a replacement.
    """
    with caplog.at_level("WARNING"):
        _stt(vad=None, **{name: "whatever"})

    assert name in caplog.text
    assert speechmatics_stt._DROPPED_ARGS[name] in caplog.text


def test_migrated_kwargs_carry_over(caplog) -> None:
    """A renamed argument keeps working, and the explicit new name always wins."""
    with caplog.at_level("WARNING"):
        instance = _stt(vad=None, speaker_active_format="@{speaker_id}: {text}")

    assert instance._stt_options.speaker_format == "@{speaker_id}: {text}"
    assert "migrated to `speaker_format`" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        instance = _stt(vad=None, speaker_format="{text}", speaker_active_format="ignored")

    assert instance._stt_options.speaker_format == "{text}"
    assert "using `speaker_format`" in caplog.text


def test_unrecognized_kwargs_are_reported(caplog) -> None:
    """`**kwargs` swallows anything, so a typo used to cost English-by-default in silence."""
    with caplog.at_level("WARNING"):
        instance = _stt(vad=None, langauge="fr")

    assert instance._stt_options.language == "en"
    assert "langauge" in caplog.text
    assert "unrecognized" in caplog.text


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


def test_old_style_usage_runs(caplog) -> None:
    """The capstone: the shape the pre-migration README's own examples taught.

    Every line of it used something this migration removed, and all of it now constructs.
    """
    usage = """
from livekit.plugins.speechmatics import (
    STT,
    OperatingPoint,
    SpeakerFocusMode,
    TurnDetectionMode,
)

stt = STT(
    api_key="test-key",
    vad=None,
    operating_point=OperatingPoint.ENHANCED,
    turn_detection_mode=TurnDetectionMode.ADAPTIVE,
    focus_mode=SpeakerFocusMode.RETAIN,
)
stt.update_speakers(focus_speakers=["S1"])
"""
    namespace: dict[str, Any] = {}

    with caplog.at_level("WARNING"):
        exec(usage, namespace)

    assert namespace["stt"].model == DEFAULT_MODEL.value
    for deprecated in (OPERATING_POINT, SPEAKER_FOCUS_MODE, "ADAPTIVE", "focus_mode"):
        assert deprecated in caplog.text
