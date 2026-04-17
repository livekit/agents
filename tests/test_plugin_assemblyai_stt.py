"""Tests for AssemblyAI STT plugin configuration options."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import pytest

from livekit.agents import stt as base_stt
from livekit.agents.types import NOT_GIVEN


async def test_vad_threshold_default():
    """Test vad_threshold is not set by default."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt._opts.vad_threshold is NOT_GIVEN


async def test_vad_threshold_set():
    """Test vad_threshold can be set in constructor."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", vad_threshold=0.3)
    assert stt._opts.vad_threshold == 0.3


async def test_vad_threshold_boundary_values():
    """Test vad_threshold accepts boundary values (0 and 1)."""
    from livekit.plugins.assemblyai import STT

    stt_low = STT(api_key="test-key", vad_threshold=0.0)
    assert stt_low._opts.vad_threshold == 0.0

    stt_high = STT(api_key="test-key", vad_threshold=1.0)
    assert stt_high._opts.vad_threshold == 1.0


async def test_vad_threshold_update():
    """Test vad_threshold can be updated dynamically."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", vad_threshold=0.5)
    stt.update_options(vad_threshold=0.7)
    assert stt._opts.vad_threshold == 0.7


async def test_vad_threshold_update_from_default():
    """Test vad_threshold can be set via update_options when not initially set."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt._opts.vad_threshold is NOT_GIVEN

    stt.update_options(vad_threshold=0.4)
    assert stt._opts.vad_threshold == 0.4


async def test_vad_threshold_with_other_options():
    """Test vad_threshold works alongside other options."""
    from livekit.plugins.assemblyai import STT

    stt = STT(
        api_key="test-key",
        vad_threshold=0.6,
        end_of_turn_confidence_threshold=0.8,
        max_turn_silence=1000,
    )
    assert stt._opts.vad_threshold == 0.6
    assert stt._opts.end_of_turn_confidence_threshold == 0.8
    assert stt._opts.max_turn_silence == 1000


async def test_vad_threshold_partial_update():
    """Test updating vad_threshold doesn't affect other options."""
    from livekit.plugins.assemblyai import STT

    stt = STT(
        api_key="test-key",
        vad_threshold=0.5,
        max_turn_silence=500,
    )
    stt.update_options(vad_threshold=0.8)

    assert stt._opts.vad_threshold == 0.8
    assert stt._opts.max_turn_silence == 500


# ---------------------------------------------------------------------------
# filter_disfluencies — config
# ---------------------------------------------------------------------------


async def test_filter_disfluencies_default_off():
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt._opts.filter_disfluencies is False


async def test_filter_disfluencies_set_true():
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", filter_disfluencies=True)
    assert stt._opts.filter_disfluencies is True


async def test_filter_disfluencies_update():
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    stt.update_options(filter_disfluencies=True)
    assert stt._opts.filter_disfluencies is True

    stt.update_options(filter_disfluencies=False)
    assert stt._opts.filter_disfluencies is False


# ---------------------------------------------------------------------------
# filter_disfluencies — pure-function helpers
# ---------------------------------------------------------------------------


def test_is_pure_disfluency_single_tokens():
    from livekit.plugins.assemblyai.stt import _is_pure_disfluency

    for token in (
        "hm",
        "hmm",
        "mm",
        "mmh",
        "mmhm",
        "mmhmm",
        "mhmm",
        "mm-hm",
        "mm-hmm",
        "mhm",
        "huh",
        "uh",
        "uh-huh",
        "uh-uh",
        "um",
        "umm",
        "uhm",
        "er",
        "erm",
        "ah",
        "oh",
        "eh",
    ):
        assert _is_pure_disfluency(token) is True


def test_is_pure_disfluency_case_insensitive():
    from livekit.plugins.assemblyai.stt import _is_pure_disfluency

    assert _is_pure_disfluency("MM") is True
    assert _is_pure_disfluency("Uh-Huh") is True
    assert _is_pure_disfluency("HUH") is True


def test_is_pure_disfluency_trailing_punctuation():
    from livekit.plugins.assemblyai.stt import _is_pure_disfluency

    assert _is_pure_disfluency("Mm.") is True
    assert _is_pure_disfluency("uh-huh?") is True
    assert _is_pure_disfluency("mm!") is True
    # Real-world regression: AssemblyAI emits hyphenated form with trailing period
    assert _is_pure_disfluency("Mm-hmm.") is True
    assert _is_pure_disfluency("Mm-hmm,") is True


def test_is_pure_disfluency_multi_token():
    from livekit.plugins.assemblyai.stt import _is_pure_disfluency

    assert _is_pure_disfluency("mm mm") is True
    assert _is_pure_disfluency("mm hm") is True
    assert _is_pure_disfluency("uh-huh mm") is True


def test_is_pure_disfluency_mixed_or_empty():
    from livekit.plugins.assemblyai.stt import _is_pure_disfluency

    assert _is_pure_disfluency("mm hello") is False
    assert _is_pure_disfluency("hello there") is False
    assert _is_pure_disfluency("516") is False
    assert _is_pure_disfluency("") is False
    assert _is_pure_disfluency("   ") is False


def test_strip_em_dashes():
    from livekit.plugins.assemblyai.stt import _strip_em_dashes

    assert _strip_em_dashes("516\u2014") == "516 "
    assert _strip_em_dashes("hello\u2014world") == "hello world"
    assert _strip_em_dashes("no dashes here") == "no dashes here"
    assert _strip_em_dashes("") == ""


def test_strip_em_dashes_leaves_en_dash_alone():
    from livekit.plugins.assemblyai.stt import _strip_em_dashes

    assert _strip_em_dashes("1\u20132") == "1\u20132"


# ---------------------------------------------------------------------------
# filter_disfluencies — integration with _process_stream_event
# ---------------------------------------------------------------------------


async def _noop_run(self: Any) -> None:
    await asyncio.Event().wait()


class _CaptureChan:
    """Stand-in for SpeechStream._event_ch that records events for assertions."""

    def __init__(self) -> None:
        self.events: list[base_stt.SpeechEvent] = []

    def send_nowait(self, item: base_stt.SpeechEvent) -> None:
        self.events.append(item)

    def close(self) -> None:
        pass


@pytest.fixture
def _patch_ws(monkeypatch):
    from livekit.plugins.assemblyai import stt as aai_stt

    monkeypatch.setattr(aai_stt.SpeechStream, "_run", _noop_run)


@pytest.fixture
async def _http_session():
    session = aiohttp.ClientSession()
    try:
        yield session
    finally:
        await session.close()


def _make_stt(http_session: aiohttp.ClientSession, **kwargs) -> Any:
    from livekit.plugins.assemblyai import STT

    return STT(api_key="test-key", http_session=http_session, **kwargs)


def _make_stream(stt_instance) -> tuple[Any, _CaptureChan]:
    stream = stt_instance.stream()
    cap = _CaptureChan()
    stream._event_ch = cap
    return stream, cap


async def _cleanup(stream: Any) -> None:
    for task in (getattr(stream, "_task", None), getattr(stream, "_metrics_task", None)):
        if task is None:
            continue
        task.cancel()
        try:
            await task
        except BaseException:
            pass


def _turn(
    *,
    transcript: str = "",
    utterance: str = "",
    words: list[dict] | None = None,
    end_of_turn: bool = False,
) -> dict:
    return {
        "type": "Turn",
        "transcript": transcript,
        "utterance": utterance,
        "words": words or [],
        "end_of_turn": end_of_turn,
        "language_code": "en",
    }


async def test_filter_off_passes_disfluency_through(_patch_ws, _http_session):
    """With filter off (default), disfluency transcripts reach downstream."""
    stt = _make_stt(_http_session)  # filter disabled
    stream, cap = _make_stream(stt)
    try:
        stream._process_stream_event(
            _turn(
                transcript="mm",
                words=[{"text": "mm", "start": 0, "end": 500, "confidence": 0.9}],
                end_of_turn=True,
            )
        )

        finals = [e for e in cap.events if e.type == base_stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 1
        assert finals[0].alternatives[0].text == "mm"
    finally:
        await _cleanup(stream)


async def test_filter_on_suppresses_interim_and_preflight_when_pure_disfluency(
    _patch_ws, _http_session
):
    """With filter on, pure-disfluency interim/preflight events are not emitted."""
    stt = _make_stt(_http_session, filter_disfluencies=True)
    stream, cap = _make_stream(stt)
    try:
        stream._process_stream_event(
            _turn(
                transcript="mm",
                utterance="mm",
                words=[{"text": "mm", "start": 0, "end": 500, "confidence": 0.9}],
                end_of_turn=False,
            )
        )

        types = [e.type for e in cap.events]
        assert base_stt.SpeechEventType.INTERIM_TRANSCRIPT not in types
        assert base_stt.SpeechEventType.PREFLIGHT_TRANSCRIPT not in types
        # no end_of_turn, so no FINAL
        assert base_stt.SpeechEventType.FINAL_TRANSCRIPT not in types
    finally:
        await _cleanup(stream)


async def test_filter_on_emits_empty_final_for_pure_disfluency(_patch_ws, _http_session):
    """Pure-disfluency final transcript is emitted with empty text (preserves
    _final_transcript_received signal in AudioRecognition) and END_OF_SPEECH still fires."""
    stt = _make_stt(_http_session, filter_disfluencies=True)
    stream, cap = _make_stream(stt)
    try:
        stream._process_stream_event(
            _turn(
                transcript="mm",
                words=[{"text": "mm", "start": 0, "end": 500, "confidence": 0.9}],
                end_of_turn=True,
            )
        )

        finals = [e for e in cap.events if e.type == base_stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 1
        assert finals[0].alternatives[0].text == ""

        eos = [e for e in cap.events if e.type == base_stt.SpeechEventType.END_OF_SPEECH]
        assert len(eos) == 1
    finally:
        await _cleanup(stream)


async def test_filter_on_mixed_transcript_passes_through(_patch_ws, _http_session):
    """Mixed content (disfluency + real word) is not filtered."""
    stt = _make_stt(_http_session, filter_disfluencies=True)
    stream, cap = _make_stream(stt)
    try:
        stream._process_stream_event(
            _turn(
                transcript="mm hello",
                words=[
                    {"text": "mm", "start": 0, "end": 200, "confidence": 0.9},
                    {"text": "hello", "start": 200, "end": 800, "confidence": 0.95},
                ],
                end_of_turn=True,
            )
        )

        finals = [e for e in cap.events if e.type == base_stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 1
        assert finals[0].alternatives[0].text == "mm hello"
    finally:
        await _cleanup(stream)


async def test_filter_on_strips_em_dashes_from_transcript_and_words(_patch_ws, _http_session):
    """Em-dashes are replaced with a space in transcript and each word.text."""
    stt = _make_stt(_http_session, filter_disfluencies=True)
    stream, cap = _make_stream(stt)
    try:
        stream._process_stream_event(
            _turn(
                transcript="516\u2014",
                words=[{"text": "516\u2014", "start": 0, "end": 500, "confidence": 0.9}],
                end_of_turn=True,
            )
        )

        finals = [e for e in cap.events if e.type == base_stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 1
        assert finals[0].alternatives[0].text == "516 "
        words = finals[0].alternatives[0].words or []
        # TimedString subclasses str
        assert [str(w) for w in words] == ["516 "]
    finally:
        await _cleanup(stream)


async def test_filter_on_multi_word_disfluency_is_filtered(_patch_ws, _http_session):
    """'mm mm' is suppressed as pure disfluency."""
    stt = _make_stt(_http_session, filter_disfluencies=True)
    stream, cap = _make_stream(stt)
    try:
        stream._process_stream_event(
            _turn(
                transcript="mm mm",
                words=[
                    {"text": "mm", "start": 0, "end": 200, "confidence": 0.9},
                    {"text": "mm", "start": 200, "end": 400, "confidence": 0.9},
                ],
                end_of_turn=True,
            )
        )

        types = [e.type for e in cap.events]
        assert base_stt.SpeechEventType.INTERIM_TRANSCRIPT not in types
        finals = [e for e in cap.events if e.type == base_stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 1
        assert finals[0].alternatives[0].text == ""
    finally:
        await _cleanup(stream)
