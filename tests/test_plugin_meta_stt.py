"""Tests for the Meta Muse Voice Transcribe STT plugin: realtime event mapping.

Covers:
- The turn ordering trap: Muse emits `speechEnd` *before* `speechComplete`, but
  `AudioRecognition` runs end-of-turn detection the moment it sees `END_OF_SPEECH`,
  against whatever transcript has landed. The plugin holds the boundary and releases
  it after the final.
- CUMULATIVE partials replace rather than concatenate, and the turn's text comes from
  `speechComplete` (which the model may post-process) rather than the last partial.
- Handshake construction, including `languageBias` name mapping and keyword merging.
- Framework-managed keyterms reaching live streams, since Muse only accepts keywords
  in the connection handshake.

Mirrors the unit-test style of `tests/test_plugin_assemblyai_stt.py`: patches
`asyncio.create_task` during `SpeechStream.__init__` so no real connection is opened,
then drives `_process_stream_event` directly with canned server frames.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents import APIStatusError
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.agents.utils.aio.channel import ChanEmpty
from livekit.plugins import meta
from livekit.plugins.meta.models import to_language_bias

pytestmark = pytest.mark.plugin("meta")


def _make_stream(stt: meta.STT | None = None) -> tuple[meta.STT, meta.SpeechStream]:
    """Build a real stream without letting `_main_task` open a socket."""
    stt = stt or meta.STT(api_key="test-key", mode="ENDPOINTING", language="en-US")
    stt._session = MagicMock()

    def _fake_create_task(coro: Any, **kwargs: Any) -> MagicMock:
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt.stream()

    return stt, stream


def _drain(stream: meta.SpeechStream) -> list[SpeechEvent]:
    events: list[SpeechEvent] = []
    while True:
        try:
            events.append(stream._event_ch.recv_nowait())
        except ChanEmpty:
            return events


def _feed(frames: list[dict[str, Any]], stt: meta.STT | None = None) -> list[SpeechEvent]:
    _, stream = _make_stream(stt)
    for frame in frames:
        stream._process_stream_event(frame)
    return _drain(stream)


# ---------------------------------------------------------------------------
# Turn ordering
# ---------------------------------------------------------------------------


def test_end_of_speech_is_held_until_after_the_final():
    """`speechEnd` lands before `speechComplete` on the wire; END_OF_SPEECH must not.

    `AudioRecognition._on_stt_event` starts end-of-turn detection on END_OF_SPEECH
    using the transcript accumulated so far, so passing the frames through in wire
    order would judge the turn before its text arrived.
    """
    events = _feed(
        [
            {"type": "speechStart", "turnId": "t1", "audioProcessedMs": 400},
            {"type": "transcript", "transcript": "how is the", "final": False},
            {"type": "transcript", "transcript": "how is the weather", "final": False},
            {"type": "speechEnd", "turnId": "t1", "audioProcessedMs": 3200},
            {"type": "speechComplete", "turnId": "t1", "transcript": "How is the weather?"},
        ]
    )

    assert [e.type for e in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]


def test_speech_start_releases_a_stranded_boundary():
    """A turn whose `speechComplete` never arrives must not strand the held boundary."""
    events = _feed(
        [
            {"type": "speechEnd", "turnId": "t1"},
            {"type": "speechStart", "turnId": "t2"},
        ]
    )

    assert [e.type for e in events] == [
        SpeechEventType.END_OF_SPEECH,
        SpeechEventType.START_OF_SPEECH,
    ]


def test_speech_complete_without_a_boundary_still_ends_the_turn():
    events = _feed([{"type": "speechComplete", "transcript": "hello"}])

    assert [e.type for e in events] == [
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------


def test_final_text_comes_from_speech_complete_not_the_last_partial():
    """The model may post-process a turn between its boundary and its completion."""
    events = _feed(
        [
            {"type": "transcript", "transcript": "soc two", "final": True},
            {"type": "speechComplete", "transcript": "SOC 2."},
        ]
    )

    assert events[0].type == SpeechEventType.PREFLIGHT_TRANSCRIPT
    assert events[0].alternatives[0].text == "soc two"
    final = next(e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT)
    assert final.alternatives[0].text == "SOC 2."


def test_cumulative_partials_are_emitted_whole():
    events = _feed(
        [
            {"type": "transcript", "transcript": text, "final": False}
            for text in ("show", "show me", "show me the dashboard")
        ]
    )

    assert [e.alternatives[0].text for e in events] == [
        "show",
        "show me",
        "show me the dashboard",
    ]


def test_empty_partials_are_dropped_but_an_empty_final_still_commits():
    events = _feed(
        [
            {"type": "transcript", "transcript": "", "final": False},
            {"type": "speechComplete", "transcript": ""},
        ]
    )

    assert [e.type for e in events] == [
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert events[0].alternatives[0].text == ""


def test_speaker_label_latches_onto_following_transcripts():
    events = _feed(
        [
            {"type": "speaker", "label": "A"},
            {"type": "transcript", "transcript": "hi", "final": False},
        ]
    )

    assert events[0].alternatives[0].speaker_id == "A"


def test_speech_start_time_is_anchored_on_the_connection():
    """`audioProcessedMs` is measured from the socket, which outlives no reconnect."""
    _, stream = _make_stream()
    stream._connected_at = 1_700_000_000.0
    stream._process_stream_event({"type": "speechStart", "audioProcessedMs": 2500})

    assert _drain(stream)[0].speech_start_time == 1_700_000_002.5


def test_error_frame_raises_api_status_error():
    with pytest.raises(APIStatusError):
        _feed([{"type": "error", "message": "quota exceeded"}])


def test_unknown_frames_are_ignored():
    assert (
        _feed([{"type": "audioProgress", "audioProcessedMs": 100}, {"type": "somethingNew"}]) == []
    )


# ---------------------------------------------------------------------------
# Handshake and options
# ---------------------------------------------------------------------------


def test_handshake_carries_every_documented_field():
    stt = meta.STT(
        api_key="KEY",
        mode="DIARIZATION",
        language=["en", "es"],
        keywords=["Muse"],
        zdr_override=True,
    )

    assert stt._opts.handshake("KEY") == {
        "authorization": {"accessToken": "Bearer KEY"},
        "audioEncoding": "PCM_24KHZ",
        "model": "muse-voice-transcribe-1.0",
        "mode": "DIARIZATION",
        "partialMode": "CUMULATIVE",
        "emitAudioProgress": False,
        "keywords": ["Muse"],
        "languageBias": ["English", "Spanish"],
        "zdrOverride": True,
    }


def test_language_bias_maps_codes_to_names_and_drops_the_unsupported():
    """Muse takes English language names, not BCP-47 codes, and rejects unknown ones."""
    assert to_language_bias(["en-US", "fr", "cmn-Hans-CN", "fil", "xx", "en"]) == [
        "English",
        "French",
        "Mandarin Chinese",
        "Tagalog",
    ]


def test_session_keyterms_reach_a_live_stream():
    """Keywords are handshake-only, so an update has to reach the stream's own copy."""
    stt, stream = _make_stream(meta.STT(api_key="test-key"))
    stt._update_session_keyterms(["LiveKit", "Muse"])

    assert stream._reconnect_event.is_set()
    assert stream._opts.handshake("K")["keywords"] == ["LiveKit", "Muse"]


def test_session_keyterms_do_not_clobber_caller_keywords():
    stt = meta.STT(api_key="test-key", keywords=["Agents"])
    stt._update_session_keyterms(["Agents", "Muse"])

    assert stt._opts.handshake("K")["keywords"] == ["Agents", "Muse"]
    assert stt._opts.keywords == ["Agents"]


def test_update_options_keeps_an_explicit_per_stream_language():
    """`stream(language=...)` speaks for one stream; a recognizer-wide update does not."""
    stt = meta.STT(api_key="test-key", language="en")
    stt._session = MagicMock()

    def _fake_create_task(coro: Any, **kwargs: Any) -> MagicMock:
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt.stream(language="fr")

    stt.update_options(language="de", keywords=["Muse"])

    assert stream._opts.handshake("K")["languageBias"] == ["French"]
    assert stream._opts.handshake("K")["keywords"] == ["Muse"]


def test_update_options_reaches_a_stream_without_its_own_language():
    """A stream that never asked for a language follows the recognizer."""
    stt, stream = _make_stream(meta.STT(api_key="test-key", language="en"))

    stt.update_options(language="de")

    assert stream._opts.handshake("K")["languageBias"] == ["German"]


def test_mode_update_moves_the_diarization_capability():
    stt = meta.STT(api_key="test-key", mode="ENDPOINTING")
    assert stt.capabilities.diarization is False

    stt.update_options(mode="DIARIZATION")

    assert stt.capabilities.diarization is True


# ---------------------------------------------------------------------------
# Reconnect scheduling and shutdown
# ---------------------------------------------------------------------------


def test_a_reconnect_requested_mid_turn_waits_for_the_boundary():
    """Reopening the socket mid-utterance loses the turn's final, so it is deferred."""
    stt, stream = _make_stream()
    stream._process_stream_event({"type": "speechStart", "turnId": "t1"})

    stt._update_session_keyterms(["Muse"])
    assert stream._reconnect_event.is_set() is False

    stream._process_stream_event({"type": "speechComplete", "transcript": "hello"})
    assert stream._reconnect_event.is_set() is True


def test_a_reconnect_requested_between_turns_fires_immediately():
    stt, stream = _make_stream()

    stt._update_session_keyterms(["Muse"])

    assert stream._reconnect_event.is_set() is True


async def test_aclose_closes_the_streams_it_handed_out():
    """The base implementation is a no-op, which would leave stream tasks running."""
    stt, stream = _make_stream()
    closed: list[bool] = []

    async def _fake_aclose() -> None:
        closed.append(True)

    stream.aclose = _fake_aclose  # type: ignore[method-assign]

    await stt.aclose()

    assert closed == [True]
    assert len(stt._streams) == 0


async def test_aclose_keeps_going_when_one_stream_fails():
    stt, first = _make_stream()
    _, second = _make_stream(stt)
    closed: list[str] = []

    async def _boom() -> None:
        raise RuntimeError("already gone")

    async def _ok() -> None:
        closed.append("second")

    first.aclose = _boom  # type: ignore[method-assign]
    second.aclose = _ok  # type: ignore[method-assign]

    await stt.aclose()

    assert closed == ["second"]


def test_diarization_mode_advertises_the_capability():
    assert meta.STT(api_key="k", mode="DIARIZATION").capabilities.diarization is True
    assert meta.STT(api_key="k", mode="ENDPOINTING").capabilities.diarization is False


def test_delta_partials_are_rejected_on_the_realtime_stream():
    with pytest.raises(ValueError, match="DELTA"):
        meta.STT(api_key="k", partial_mode="DELTA")


def test_encoding_selects_the_sample_rate():
    assert meta.STT(api_key="k", encoding="PCM_16KHZ")._opts.sample_rate == 16000
    assert meta.STT(api_key="k", encoding="PCM_24KHZ")._opts.sample_rate == 24000


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    with pytest.raises(ValueError, match="META_API_KEY"):
        meta.STT()
