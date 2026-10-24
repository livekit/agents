"""Tests for the Keyframe plugin: timed emotion cues for persona-2.0-live.

Covers the framed audio stream (``_framing``, ``KeyframeAudioOutput``), the timed
``set_emotion`` on top of it, and ``emotion_tts_node``: the marker parser, the cue
planner that times markers against TTS word timings, and the node that ties them to
``Agent.default.tts_node``. The room and stream writer are faked; ``emotion_tts_node``
runs in a real ``AgentSession`` with a fake TTS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import struct
from collections.abc import AsyncIterable, AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit import rtc
from livekit.agents import AgentSession, tts
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    USERDATA_TIMED_TRANSCRIPT,
    APIConnectOptions,
    TimedString,
)
from livekit.agents.voice import Agent, ModelSettings
from livekit.plugins.keyframe import (
    EMOTIONS,
    USERDATA_EMOTION_CUES,
    EmotionCue,
    _framing,
    _output,
    _tags,
    _tts_node,
    emotion_instructions,
    emotion_tts_node,
)

from .fake_llm import FakeLLM
from .fake_tts import FakeChunkedStream, FakeSynthesizeStream, FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

SR = _output.SAMPLE_RATE
AVATAR = "keyframe-avatar-agent"

_HEADER = struct.Struct("<BI")


def _split_records(data: bytes) -> list[tuple[int, bytes]]:
    """Split a write into (record type, payload) the way the avatar does."""
    out: list[tuple[int, bytes]] = []
    offset = 0
    while offset < len(data):
        kind, length = _HEADER.unpack_from(data, offset)
        offset += _HEADER.size
        out.append((kind, data[offset : offset + length]))
        offset += length
    assert offset == len(data), "a write must hold whole records"
    return out


def _records(data: bytes) -> list[tuple[str, Any]]:
    """The framed records of a write: ("audio", n_bytes) or ("cue", body)."""
    out: list[tuple[str, Any]] = []
    for kind, payload in _split_records(data):
        if kind == _framing.RECORD_AUDIO:
            out.append(("audio", len(payload)))
        elif kind == _framing.RECORD_CUE:
            out.append(("cue", json.loads(payload)))
        else:
            raise AssertionError(f"unknown record type {kind:#x}")
    return out


def _frame(
    ms: int, *, sample_rate: int = SR, channels: int = 1, pcm: bytes | None = None
) -> rtc.AudioFrame:
    n = sample_rate * ms // 1000
    return rtc.AudioFrame(
        data=pcm if pcm is not None else b"\x00\x00" * (n * channels),
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=n,
    )


def _ramp(n: int) -> bytes:
    """``n`` PCM16 samples that all differ, so reordered or altered audio shows."""
    return struct.pack(f"<{n}h", *range(n))


def _with_cues(frame: rtc.AudioFrame, *cues: EmotionCue) -> rtc.AudioFrame:
    frame.userdata[USERDATA_EMOTION_CUES] = list(cues)
    return frame


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------


def test_framing_records_match_the_spec() -> None:
    cue = _framing.encode_cue("happy", 57600)
    kind, length = _HEADER.unpack_from(cue)
    assert kind == 0x02
    assert length == len(cue) - _HEADER.size
    assert json.loads(cue[_HEADER.size :]) == {"emotion": "happy", "at": 57600}

    pcm = b"\x01\x02" * 480
    audio = _framing.encode_audio(pcm)
    assert audio[:1] == b"\x01"
    assert struct.unpack_from("<I", audio, 1)[0] == len(pcm)
    assert audio[_HEADER.size :] == pcm

    assert _records(cue + audio) == [("cue", {"emotion": "happy", "at": 57600}), ("audio", 960)]


# ---------------------------------------------------------------------------
# KeyframeAudioOutput
# ---------------------------------------------------------------------------


class _Room:
    """A connected room with the avatar present, a recording stream writer, and a
    recording ``publish_data``."""

    def __init__(self) -> None:
        self.mock = MagicMock()
        self.mock.isconnected.return_value = True
        self.mock.local_participant._rpc_handlers = {}
        self.avatar = SimpleNamespace(identity=AVATAR, attributes={})
        self.mock.remote_participants = {AVATAR: self.avatar}
        self.writer = MagicMock()
        self.writer.write = AsyncMock()
        self.writer.aclose = AsyncMock()
        self.mock.local_participant.stream_bytes = AsyncMock(return_value=self.writer)
        self.mock.local_participant.publish_data = AsyncMock()

    @property
    def stream_attributes(self) -> dict[str, str]:
        kwargs: dict[str, Any] = self.mock.local_participant.stream_bytes.call_args.kwargs
        attrs: dict[str, str] = kwargs["attributes"]
        return attrs

    @property
    def writes(self) -> list[bytes]:
        return [c.args[0] for c in self.writer.write.call_args_list]

    @property
    def framed_writes(self) -> list[list[tuple[str, Any]]]:
        return [_records(w) for w in self.writes]

    @property
    def audio(self) -> bytes:
        """The PCM of every framed audio record, in order."""
        return b"".join(
            payload
            for w in self.writes
            for kind, payload in _split_records(w)
            if kind == _framing.RECORD_AUDIO
        )

    @property
    def emotions_published(self) -> list[tuple[bytes, bool]]:
        return [
            (c.args[0], c.kwargs["reliable"])
            for c in self.mock.local_participant.publish_data.call_args_list
            if c.kwargs["topic"] == _output.CONTROL_TOPIC
        ]


async def _output_for(room: _Room, *, framed: bool = True) -> _output.KeyframeAudioOutput:
    out = _output.KeyframeAudioOutput(room.mock, destination_identity=AVATAR, framed=framed)
    # the avatar has joined and published video: skip DataStreamAudioOutput's wait
    out._started = True
    out._start_atask = asyncio.create_task(asyncio.sleep(0))
    return out


async def _settle(out: _output.KeyframeAudioOutput) -> None:
    await asyncio.gather(*out._tasks)


async def test_output_frames_the_stream() -> None:
    room = _Room()
    out = await _output_for(room)

    pcm = _ramp(SR // 10)
    await out.capture_frame(_frame(100, pcm=pcm))
    assert room.stream_attributes == {
        "sample_rate": "24000",
        "num_channels": "1",
        "kfl.framing": "1",
    }
    # 24 kHz mono needs no resampler, so the flush adds no tail record
    out.flush()
    await _settle(out)
    assert room.framed_writes == [[("audio", 4800)]]
    assert room.audio == pcm
    assert room.writer.aclose.await_count == 1


async def test_output_without_framing_sends_raw_pcm_and_ignores_cues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    room = _Room()
    out = await _output_for(room, framed=False)

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.keyframe"):
        for _ in range(2):
            await out.capture_frame(_with_cues(_frame(100), EmotionCue("happy")))
    assert "kfl.framing" not in room.stream_attributes
    assert room.writes == [b"\x00\x00" * 2400] * 2
    assert room.emotions_published == []
    assert len([r for r in caplog.records if "ignoring emotion cues" in r.message]) == 1


async def test_output_times_cues_against_the_segment() -> None:
    room = _Room()
    out = await _output_for(room)

    await out.capture_frame(_with_cues(_frame(100), EmotionCue("happy"), EmotionCue("sad", 0.05)))
    await out.capture_frame(_with_cues(_frame(100), EmotionCue("angry", 0.02)))
    assert room.framed_writes == [
        [
            ("cue", {"emotion": "happy", "at": 0}),
            ("cue", {"emotion": "sad", "at": round(0.05 * SR)}),
            ("audio", 4800),
        ],
        # the second frame starts 100 ms into the segment
        [("cue", {"emotion": "angry", "at": round(0.12 * SR)}), ("audio", 4800)],
    ]


async def test_output_times_cues_in_input_seconds_when_resampling() -> None:
    room = _Room()
    out = await _output_for(room)

    # the resampler holds back part of the first frame, so less than 100 ms comes out of it
    await out.capture_frame(_frame(100, sample_rate=16000))
    await out.capture_frame(_with_cues(_frame(100, sample_rate=16000), EmotionCue("angry", 0.02)))
    out.flush()
    await _settle(out)

    first, second, tail = room.framed_writes
    assert first[0][0] == "audio" and first[0][1] < 4800
    # cue offsets are input seconds: 0.12 s of the stream, whatever the resampler emitted
    assert second[0] == ("cue", {"emotion": "angry", "at": round(0.12 * SR)})
    # the held-back audio goes out as a final record, then the stream closes
    assert [kind for kind, _ in tail] == ["audio"]
    assert [name for name, *_ in room.writer.mock_calls][-2:] == ["write", "aclose"]
    assert out._stream_writer is None


async def test_output_flushes_the_resampler_when_the_input_format_changes() -> None:
    room = _Room()
    out = await _output_for(room)

    # 16 kHz, then 48 kHz, then 24 kHz in one segment: each switch first drains the
    # resampler it retires, so no audio is lost and the cue clock stays on the PCM
    await out.capture_frame(_frame(100, sample_rate=16000))
    await out.capture_frame(_frame(100, sample_rate=48000))
    await out.capture_frame(_with_cues(_frame(100), EmotionCue("happy")))
    out.flush()
    await _settle(out)

    assert len(room.audio) == 3 * 2 * SR // 10  # 300 ms at 24 kHz, nothing dropped
    assert ("cue", {"emotion": "happy", "at": round(0.2 * SR)}) in room.framed_writes[2]
    assert out._resampler is None


async def test_output_downmixes_stereo() -> None:
    room = _Room()
    out = await _output_for(room)

    n = SR // 10
    await out.capture_frame(_frame(100, channels=2, pcm=struct.pack("<2h", 1000, 3000) * n))
    assert room.audio == struct.pack("<h", 2000) * n


async def test_set_emotion_is_a_reliable_now_message() -> None:
    room = _Room()
    room.avatar.attributes["kfl.protocol"] = "2"
    out = await _output_for(room)

    # idle, mid-segment and after the segment closes alike: a control message the
    # avatar applies from its next frame, never a cue in the stream
    await out.set_emotion("happy")
    await out.capture_frame(_frame(100))
    await out.set_emotion("sad")
    out.flush()
    await _settle(out)
    await out.set_emotion("angry")
    assert room.emotions_published == [
        (b"emotion_now:happy", True),
        (b"emotion_now:sad", True),
        (b"emotion_now:angry", True),
    ]
    assert room.framed_writes == [[("audio", 4800)]]


@pytest.mark.parametrize("protocol", [None, "1", "v1"])
async def test_set_emotion_sends_the_legacy_message_to_an_older_avatar(
    protocol: str | None,
) -> None:
    room = _Room()
    if protocol is not None:
        room.avatar.attributes["kfl.protocol"] = protocol
    out = await _output_for(room, framed=False)
    await out.set_emotion("neutral")
    await out.capture_frame(_frame(100))
    await out.set_emotion("happy")
    # an avatar that predates emotion_now: gets the message it understands
    assert room.emotions_published == [
        (b"emotion:neutral", True),
        (b"emotion:happy", True),
    ]
    assert len(room.writes) == 1  # only the audio


async def test_set_emotion_follows_the_protocol_the_avatar_advertises() -> None:
    room = _Room()
    out = await _output_for(room)
    # the avatar sets its attribute shortly after joining; until then, legacy
    await out.set_emotion("happy")
    room.avatar.attributes["kfl.protocol"] = "2"
    await out.set_emotion("sad")
    room.mock.remote_participants.clear()  # the avatar left
    await out.set_emotion("angry")
    assert room.emotions_published == [
        (b"emotion:happy", True),
        (b"emotion_now:sad", True),
        (b"emotion:angry", True),
    ]


async def test_output_resamples_each_segment_afresh() -> None:
    room = _Room()
    out = await _output_for(room)

    resamplers = []
    lengths = []
    for _ in range(2):
        before = len(room.audio)
        await out.capture_frame(_frame(100, sample_rate=16000, pcm=_ramp(1600)))
        resamplers.append(out._resampler)
        out.flush()
        await _settle(out)
        lengths.append(len(room.audio) - before)
    # a new resampler for each segment, dropped with it, so no state carries over. (The
    # output bytes can't show this: a reused resampler differs from a fresh one by a
    # couple of LSBs, and on Linux so do two fresh ones.)
    assert resamplers[0] is not None and resamplers[1] is not None
    assert resamplers[1] is not resamplers[0]
    assert out._resampler is None
    # and each segment came out whole: 100 ms at 24 kHz, including the flushed tail
    assert lengths == [2 * SR // 10] * 2


async def test_avatar_session_validates_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.keyframe.avatar import AvatarSession

    monkeypatch.setenv("KEYFRAME_API_KEY", "keyframe_sk_test")
    session = AvatarSession(persona_slug="public:cosmo-2.0-live-preview")
    # before start(): a warning, no crash
    await session.set_emotion("happy")

    output = MagicMock()
    output.set_emotion = AsyncMock()
    session._audio_output = output
    await session.set_emotion("happy")
    output.set_emotion.assert_awaited_once_with("happy")
    await session.set_emotion("bored")  # type: ignore[arg-type]
    assert output.set_emotion.await_count == 1

    # a persona-1.5-live model renders four expressions; the others are not sent
    session._emotions = ("neutral", "happy", "sad", "angry")
    await session.set_emotion("skeptical")
    assert output.set_emotion.await_count == 1
    await session.set_emotion("sad")
    assert output.set_emotion.await_count == 2


@pytest.mark.parametrize(
    ("model_id", "framed"),
    [
        ("persona-2.0-live-preview", True),
        ("persona-2.0-live", True),
        ("persona-1.5-live", False),
        (None, False),  # an API that predates `model_id` in the response
    ],
)
async def test_avatar_session_frames_by_model(
    monkeypatch: pytest.MonkeyPatch, model_id: str | None, framed: bool
) -> None:
    from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
    from livekit.plugins.keyframe import avatar

    response: dict[str, Any] = {"reservation_id": "r1"}
    if model_id is not None:
        response["model_id"] = model_id

    class _FakeAPI:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeAPI:
            return self

        async def __aexit__(self, *_: Any) -> None:
            pass

        async def create_plugin_session(self, **_: Any) -> dict[str, Any]:
            return response

    output_cls = MagicMock()
    monkeypatch.setattr(BaseAvatarSession, "start", AsyncMock())
    monkeypatch.setattr(avatar, "KeyframeAPI", _FakeAPI)
    monkeypatch.setattr(avatar, "KeyframeAudioOutput", output_cls)
    monkeypatch.setattr(
        avatar, "get_job_context", lambda: SimpleNamespace(local_participant_identity="agent")
    )

    session = avatar.AvatarSession(persona_slug="public:cosmo", api_key="keyframe_sk_test")
    room = MagicMock()
    room.name = "room"
    await session.start(
        MagicMock(),
        room,
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key="key",
        livekit_api_secret="secret",
    )
    assert output_cls.call_args.kwargs["framed"] is framed
    # only persona-1.5-live narrows set_emotion to its four expressions; an unreported
    # model keeps the full set
    legacy = model_id is not None and not framed
    assert session._emotions == (("neutral", "happy", "sad", "angry") if legacy else EMOTIONS)


# ---------------------------------------------------------------------------
# marker parser
# ---------------------------------------------------------------------------


def _parse(chunks: list[str], **kwargs: Any) -> tuple[str, list[_tags.Marker], _tags.TagParser]:
    parser = _tags.TagParser(**kwargs)
    out, markers = "", []
    for chunk in chunks:
        text, found = parser.push(chunk)
        out += text
        markers += found
    text, found = parser.flush()
    return out + text, markers + found, parser


def test_parser_finds_expr_markers_split_across_chunks() -> None:
    chunks = [
        '<expr type="expression" la',
        'bel="happy"/> Hello there! <ex',
        'pr type="expression" label="sad"/> Bye.',
    ]
    out, markers, parser = _parse(chunks, forward_expr=True)
    assert out == "".join(chunks), "expressive on: the markup reaches the TTS"
    assert markers == [_tags.Marker(0, "happy"), _tags.Marker(len("hellothere"), "sad")]
    assert parser.spoken == "hellotherebye"

    out, markers2, _ = _parse(chunks, forward_expr=False)
    assert out == " Hello there!  Bye."
    assert markers2 == markers


def test_parser_handles_the_unclosed_form_and_other_marker_types() -> None:
    text = (
        '<expr type="expression" label="worried, concerned"> Oh '
        '<expr type="prosody" label="sad">no</expr> <expr type="break" label="1s"/> '
        '<expr type="spell">A7</expr> ok'
    )
    out, markers, parser = _parse([text], forward_expr=False)
    assert out == " Oh no  A7 ok"
    # only expression markers set an emotion, whatever another type's label says
    assert markers == [_tags.Marker(0, "concerned")]
    # words inside prosody/spell markers count as spoken
    assert parser.spoken == "ohnoa7ok"


def test_parser_finds_bracket_tags_and_always_forwards_them() -> None:
    chunks = ["[ha", "ppy] Okay. [warmly] sure [sa", "d] end"]
    out, markers, parser = _parse(chunks, style="brackets", forward_expr=False)
    assert out == "[happy] Okay. [warmly] sure [sad] end"
    assert markers == [_tags.Marker(0, "happy"), _tags.Marker(len("okaysure"), "sad")]
    assert parser.spoken == "okaysureend"


def test_parser_reads_brackets_as_text_in_the_expr_style() -> None:
    text = 'See [1] <expr type="expression" label="happy"/>[happy] there.'
    out, markers, parser = _parse([text], forward_expr=False)
    assert out == "See [1] [happy] there."
    assert markers == [_tags.Marker(len("see1"), "happy")]
    assert parser.spoken == "see1happythere"


def test_parser_reads_expr_markup_in_the_brackets_style() -> None:
    text = '<expr type="expression" label="sad"/> Oh [happy] no.'
    out, markers, _ = _parse([text], style="brackets", forward_expr=False)
    assert out == " Oh [happy] no."
    assert markers == [_tags.Marker(0, "sad"), _tags.Marker(len("oh"), "happy")]


def test_parser_rejects_an_unknown_style() -> None:
    with pytest.raises(ValueError):
        _tags.TagParser(style="other", forward_expr=False)  # type: ignore[arg-type]


def test_parser_reports_only_the_allowed_emotions() -> None:
    text = (
        '<expr type="expression" label="happy"/> hi <expr type="expression" label="sad"/> '
        'there <expr type="expression" label="happy"/>'
    )
    _, markers, _ = _parse([text], forward_expr=False, emotions=("neutral", "sad"))
    assert markers == [_tags.Marker(len("hi"), "sad")]


def test_parser_gives_up_holding_an_unclosed_expr() -> None:
    # held for good, it would keep everything after it from the TTS until the reply ends
    stray = '<expr type="expression" label="' + "x" * _tags._MAX_EXPR_HOLD
    parser = _tags.TagParser(forward_expr=False)
    assert parser.push(stray) == (stray, [])


def test_parser_treats_long_or_multiline_brackets_as_text() -> None:
    long_run = "[" + "x" * (_tags.MAX_TAG_LENGTH + 1) + "]"
    out, markers, _ = _parse([long_run, " [a\nb] [happy]"], style="brackets", forward_expr=False)
    assert out == f"{long_run} [a\nb] [happy]"
    # their letters count as spoken
    assert markers == [_tags.Marker(len("x" * 33 + "ab"), "happy")]


@pytest.mark.parametrize("text", ["hi [par", 'hi <expr type="expression" label="x"'])
def test_parser_releases_a_partial_marker_at_the_end_of_the_stream(text: str) -> None:
    parser = _tags.TagParser(style="brackets", forward_expr=False)
    assert parser.push(text) == ("hi ", []), "held: the marker may complete"
    assert parser.flush() == (text.removeprefix("hi "), [])


def test_parser_ignores_a_legacy_expression_tag() -> None:
    text = '<expression value="happy"/> hi'
    out, markers, _ = _parse([text], forward_expr=False)
    assert (out, markers) == (text, [])


def test_parser_treats_a_lone_less_than_as_text() -> None:
    parser = _tags.TagParser(forward_expr=False)
    # one mid-chunk goes straight out; one ending a chunk may still open a marker
    assert parser.push("3 < 5 and 7 <") == ("3 < 5 and 7 ", [])
    assert parser.push("8") == ("<8", [])
    assert parser.spoken == "35and78"


@pytest.mark.parametrize(
    ("label", "emotion"),
    [
        # the avatar's own names, in any case
        ("Happy", "happy"),
        ("skeptical", "skeptical"),
        # free-form (Inworld-style) labels naming one; the earliest wins
        ("sincere and concerned", "concerned"),
        ("excited, a little fast", "excited"),
        ("sad, then happy", "sad"),
        # whole words only
        ("excitedly", None),
        ("unhappy", None),
        # a word close to one names nothing: no cue
        ("sympathetic", None),
        ("", None),
    ],
)
def test_emotion_for_label(label: str, emotion: str | None) -> None:
    assert _tags.emotion_for_label(label) == emotion


@pytest.mark.parametrize(
    ("name", "emotion"),
    [
        ("happy", "happy"),
        (" Sad ", "sad"),
        # anything but an exact name is a voice-only audio tag
        ("warmly", None),
        ("happy laugh", None),
    ],
)
def test_emotion_for_tag(name: str, emotion: str | None) -> None:
    assert _tags.emotion_for_tag(name) == emotion


def test_emotion_matching_respects_the_allowed_set() -> None:
    assert _tags.emotion_for_label("happy", allowed=("neutral", "sad")) is None
    assert _tags.emotion_for_label("happy, then sad", allowed=("neutral", "sad")) == "sad"
    assert _tags.emotion_for_tag("happy", allowed=("neutral", "sad")) is None


# ---------------------------------------------------------------------------
# cue planner
# ---------------------------------------------------------------------------


def _cues(frame: rtc.AudioFrame) -> list[tuple[str, float]]:
    return [(c.emotion, round(c.offset, 3)) for c in frame.userdata.get(USERDATA_EMOTION_CUES, [])]


def _cue_times(frames: list[rtc.AudioFrame]) -> list[tuple[str, float]]:
    """Every cue as (emotion, seconds from the start of the reply)."""
    out: list[tuple[str, float]] = []
    start = 0.0
    for frame in frames:
        for cue in frame.userdata.get(USERDATA_EMOTION_CUES, []):
            out.append((cue.emotion, round(start + cue.offset, 3)))
        start += frame.duration
    return out


def _planner(
    text: str, *, aligned: bool = True, max_cue_hold: float = 0.5
) -> _tts_node._CuePlanner:
    planner = _tts_node._CuePlanner(aligned=aligned, max_cue_hold=max_cue_hold)
    planner.spoken = _parse([text], forward_expr=False)[2].spoken
    return planner


def _timed(word: str, start: float, end: float | None = None) -> TimedString:
    return TimedString(word, start_time=start, end_time=end if end is not None else start)


def _run(planner: _tts_node._CuePlanner, seconds: float) -> list[rtc.AudioFrame]:
    """Push ``seconds`` of audio through ``planner`` in 100 ms frames, then end the reply."""
    frames = []
    for _ in range(round(seconds * 10)):
        frames += planner.push_frame(_frame(100))
    return frames + planner.finish()


def test_planner_opening_marker_is_at_zero() -> None:
    planner = _planner("Hello there")
    planner.add_marker(_tags.Marker(0, "happy"))
    (frame,) = planner.push_frame(_frame(100))
    assert _cues(frame) == [("happy", 0.0)]


def test_planner_times_a_marker_at_the_next_matched_word() -> None:
    planner = _planner("Hello there. Now sad words.")
    planner.add_marker(_tags.Marker(len("hellothere"), "sad"))

    # the marker's word isn't timed yet: frames past the last timed word are held
    planner.add_timings([_timed("Hello ", 0.0, 0.3), _timed("there. ", 0.3, 0.6)])
    released = []
    for _ in range(8):  # 0.8 s of audio
        released += planner.push_frame(_frame(100))
    assert len(released) == 6, "audio up to the end of 'there' may go; the rest waits"

    planner.add_timings([_timed("Now ", 0.7, 0.9)])
    released += planner.push_frame(_frame(100))
    assert len(released) == 9
    assert _cues(released[7]) == [("sad", 0.0)]  # 0.7 s is the start of frame 7
    assert all(not _cues(f) for i, f in enumerate(released) if i != 7)


def test_planner_hold_is_capped() -> None:
    planner = _planner("Hello there. Now sad words.", max_cue_hold=0.3)
    planner.add_marker(_tags.Marker(len("hellothere"), "sad"))
    planner.add_timings([_timed("Hello ", 0.0, 0.3), _timed("there. ", 0.3, 0.6)])

    released = []
    for _ in range(12):  # 1.2 s
        released += planner.push_frame(_frame(100))
    # 0.6 s released on the timings, then everything beyond a 0.3 s hold
    assert len(released) == 9
    assert all(not _cues(f) for f in released)

    # the word arrives late, in audio already released: the cue goes on the next frame
    # with a negative offset, so it still lands at the right sample of the stream
    planner.add_timings([_timed("Now ", 0.7, 0.9)])
    late = planner.push_frame(_frame(100))  # nothing pending: the held frames go too
    assert len(late) == 4
    assert _cues(late[0]) == [("sad", -0.2)]  # 0.7 s, on the frame at 0.9 s


def test_planner_matches_short_words_only_at_the_coverage_position() -> None:
    planner = _planner("I have 2 cats, not two dogs.")
    planner.add_marker(_tags.Marker(len("ihave2cats"), "sad"))
    planner.add_timings(
        [
            _timed("I ", 0.0, 0.1),
            _timed("have ", 0.1, 0.3),
            # "2" read aloud: the text has "two" later, and matching it there would jump
            # past "cats, not" and place the cue at 0.3 s
            _timed("two ", 0.3, 0.5),
            _timed("cats, ", 0.5, 0.8),
            _timed("not ", 0.8, 1.0),
            _timed("two ", 1.0, 1.2),
            _timed("dogs.", 1.2, 1.5),
        ]
    )
    # with no word to trust after "have", the cue goes by its share of the text, 10 of 20
    assert _cue_times(_run(planner, 1.5)) == [("sad", 0.75)]


def test_planner_resyncs_on_a_long_word_after_an_unmatched_token() -> None:
    planner = _planner("I have 2 cats, seriously twenty.")
    planner.add_marker(_tags.Marker(len("ihave2cats"), "happy"))
    planner.add_marker(_tags.Marker(len("ihave2catsseriously"), "sad"))
    planner.add_timings(
        [
            _timed("I ", 0.0, 0.1),
            _timed("have ", 0.1, 0.3),
            # a normalized number doesn't match the text, and "cats" is too short to
            # match ahead of it
            _timed("two ", 0.3, 0.5),
            _timed("cats, ", 0.5, 0.8),
            # a word of 6+ characters matches ahead
            _timed("seriously ", 0.9, 1.4),
            _timed("twenty.", 1.4, 1.8),
        ]
    )
    # "happy" was in the skipped span: it resolves where coverage resumed
    assert _cue_times(_run(planner, 2.0)) == [("happy", 0.9), ("sad", 1.4)]


def test_planner_matches_a_first_word_missing_its_first_letters() -> None:
    planner = _planner('Oh, thank you. <expr type="expression" label="sad"/> But we lost.')
    planner.add_marker(_tags.Marker(len("ohthankyou"), "sad"))
    planner.add_timings(
        [
            # "Oh," with the "O" dropped: skipped, it would leave nothing short to match
            _timed("h, ", 0.1, 0.3),
            _timed("thank ", 0.3, 0.6),
            _timed("you. ", 0.6, 0.9),
            _timed("But ", 1.0, 1.2),
            _timed("we ", 1.2, 1.4),
            _timed("lost.", 1.4, 1.8),
        ]
    )
    assert _cue_times(_run(planner, 1.8)) == [("sad", 1.0)]


def test_planner_matches_a_short_last_word_at_the_end_of_the_text() -> None:
    planner = _planner("Seriously, 2 ok.")
    planner.text_done()
    planner.add_marker(_tags.Marker(len("seriously2"), "happy"))
    planner.add_timings(
        [
            _timed("Seriously, ", 0.0, 0.6),
            # "2" read aloud doesn't match the text
            _timed("two ", 0.6, 0.9),
            # too short to match ahead of coverage, but it ends the text
            _timed("ok.", 1.0, 1.3),
        ]
    )
    assert _cue_times(_run(planner, 1.3)) == [("happy", 1.0)]


def test_planner_sentence_timings_without_end_times() -> None:
    # StreamAdapter: one TimedString per sentence, raw markup included, start_time only
    text = 'Hello there. <expr type="expression" label="sad"/> Now sad words.'
    planner = _planner(text)
    planner.add_marker(_tags.Marker(len("hellothere"), "sad"))

    planner.add_timings([TimedString("Hello there. ", start_time=0.0)])
    first = planner.push_frame(_frame(400))
    assert first == [], "held: the sentence's end isn't known"
    planner.add_timings(
        [TimedString('<expr type="expression" label="sad"/> Now sad words.', start_time=0.4)]
    )
    released = planner.push_frame(_frame(400))
    assert [_cues(f) for f in released] == [[], [("sad", 0.0)]]


def test_planner_marker_the_alignment_already_passed_resolves_now() -> None:
    planner = _planner("Absolutely wonderful news")
    planner.add_timings([_timed("Absolutely ", 0.0, 0.6)])
    planner.add_marker(_tags.Marker(3, "happy"))  # inside a word already timed
    (frame,) = planner.push_frame(_frame(1000))
    assert _cues(frame) == [("happy", 0.6)]


def test_planner_without_alignment_keeps_only_the_opening_marker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    planner = _planner("Hello there. Now sad words. Then angry.", aligned=False)
    planner.add_marker(_tags.Marker(0, "happy"))
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.keyframe"):
        planner.add_marker(_tags.Marker(len("hellothere"), "sad"))
        planner.add_marker(_tags.Marker(len("hellotherenowsadwords"), "angry"))

    frames = _run(planner, 1.0)
    assert len(frames) == 10, "nothing is held without alignment"
    assert _cue_times(frames) == [("happy", 0.0)]
    # the ignored markers are reported once per reply, not once per marker
    assert len([r for r in caplog.records if "ignoring an emotion marker" in r.message]) == 1


def test_planner_finish_places_pending_markers_proportionally() -> None:
    planner = _planner("one two three four", max_cue_hold=2.0)
    planner.text_done()
    planner.add_marker(_tags.Marker(len("onetwothree"), "sad"))  # 11 of 15 chars
    planner.add_marker(_tags.Marker(len("onetwothreefour"), "happy"))  # the very end
    assert planner.push_frame(_frame(500)) == []
    assert planner.push_frame(_frame(500)) == []
    frames = planner.finish()
    assert len(frames) == 2
    assert _cues(frames[0]) == []
    # 11/15 of 1.0 s is 0.733 s, 0.233 s into the second frame; the end cue rides past it
    assert _cues(frames[1]) == [("sad", 0.233), ("happy", 0.5)]


def test_planner_keeps_the_newest_frame_while_a_marker_is_pending() -> None:
    # one frame longer than the hold cap: it would be released at once, and finish()
    # would then mark a frame the caller has already sent on. It waits instead.
    planner = _planner("one two three four", max_cue_hold=0.5)
    planner.text_done()
    planner.add_marker(_tags.Marker(len("onetwothree"), "sad"))
    assert planner.push_frame(_frame(1000)) == []
    (frame,) = planner.finish()
    assert _cues(frame) == [("sad", 0.733)]


def test_planner_keeps_the_newest_frame_for_a_cue_past_it() -> None:
    # the alignment runs ahead of the audio: "there" is timed at 0.5 s but the reply's
    # audio stops at 0.4 s, so the cue can only ride the last frame, at finish()
    planner = _planner("Hello there")
    planner.add_marker(_tags.Marker(len("hello"), "sad"))
    planner.add_timings([_timed("Hello ", 0.0, 0.5), _timed("there", 0.5, 0.9)])
    released = []
    for _ in range(4):
        released += planner.push_frame(_frame(100))
    assert len(released) == 3, "the newest frame waits: the cue may still land past it"
    (last,) = planner.finish()
    assert _cues(last) == [("sad", 0.2)]  # 0.5 s, on the frame that starts at 0.3 s


# ---------------------------------------------------------------------------
# _emotion_frames with a fake inner node
# ---------------------------------------------------------------------------


async def _chunks(*parts: str) -> AsyncIterator[str]:
    for part in parts:
        yield part


class _FakeNode:
    """A ``tts_node`` stand-in: records the text it is given and yields scripted frames."""

    def __init__(self, frames: list[rtc.AudioFrame]) -> None:
        self.frames = frames
        self.text = ""
        self.closed = False

    def __call__(self, text: AsyncIterable[str]) -> AsyncIterator[rtc.AudioFrame]:
        return self._run(text)

    async def _run(self, text: AsyncIterable[str]) -> AsyncIterator[rtc.AudioFrame]:
        try:
            async for chunk in text:
                self.text += chunk
            for frame in self.frames:
                yield frame
        finally:
            self.closed = True


def _timed_frame(ms: int, *tokens: TimedString) -> rtc.AudioFrame:
    frame = _frame(ms)
    frame.userdata[USERDATA_TIMED_TRANSCRIPT] = list(tokens)
    return frame


async def _bracket_frames(node: _FakeNode, *chunks: str) -> list[rtc.AudioFrame]:
    """``_emotion_frames`` in the brackets style, on a TTS with word timings."""
    gen = _tts_node._emotion_frames(
        _chunks(*chunks), node, style="brackets", forward_expr=False, aligned=True
    )
    return [f async for f in gen]


def _transcripts(frames: list[rtc.AudioFrame]) -> list[list[str]]:
    return [[str(t) for t in f.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])] for f in frames]


MARKED = (
    '<expr type="expression" label="happy"/> Hello ',
    'there. <expr type="expr',
    'ession" label="sad"/> Now sad words.',
)


async def test_emotion_frames_matches_word_timings_through_echoed_markup() -> None:
    # word timings for a frame's words ride on the next frame, as Cartesia delivers them,
    # and the alignment echoes the tag each marker lowered to, glued to the next word
    words = [
        ('<emotion value="happy"/>Hello ', 0.0, 0.3),
        ("there. ", 0.3, 0.6),
        ('<emotion value="sad"/>Now ', 0.7, 0.9),
        ("sad ", 0.9, 1.2),
        ("words.", 1.2, 1.6),
    ]
    tts_frames = []
    pending: list[TimedString] = []
    for i in range(17):
        tts_frames.append(_timed_frame(100, *pending))
        t = i / 10
        pending = [_timed(w, s, e) for w, s, e in words if t <= s < t + 0.1]

    node = _FakeNode(tts_frames)
    frames = [
        f
        async for f in _tts_node._emotion_frames(
            _chunks(*MARKED), node, forward_expr=True, aligned=True
        )
    ]
    assert node.text == "".join(MARKED)
    assert len(frames) == 17
    cues = {i: _cues(f) for i, f in enumerate(frames) if _cues(f)}
    assert cues == {0: [("happy", 0.0)], 7: [("sad", 0.0)]}


async def test_emotion_frames_drops_bracket_tags_from_transcripts() -> None:
    node = _FakeNode(
        [
            _timed_frame(300, _timed("[happy] ", 0.0, 0.0), _timed("Okay ", 0.0, 0.3)),
            _timed_frame(400, _timed("[sad] ", 0.3, 0.3), _timed("then.", 0.3, 0.7)),
        ]
    )
    frames = await _bracket_frames(node, "[happy] Okay [sa", "d] then.")
    assert node.text == "[happy] Okay [sad] then."
    assert [_cues(f) for f in frames] == [[("happy", 0.0)], [("sad", 0.0)]]
    assert _transcripts(frames) == [["Okay "], ["then."]]


async def test_emotion_frames_releases_a_bracket_left_open_at_the_end() -> None:
    node = _FakeNode(
        [
            _timed_frame(300, _timed("[happy] ", 0.0, 0.0), _timed("Okay, ", 0.0, 0.3)),
            _timed_frame(200, _timed("look ", 0.3, 0.4), _timed("[this ", 0.4, 0.45)),
            _timed_frame(200, _timed("over ", 0.45, 0.5), _timed("here", 0.5, 0.6)),
        ]
    )
    frames = await _bracket_frames(node, "[happy] Okay, look [this over here")
    # the words from "[" on were held back in case they opened a tag; at the end they are
    # text after all, and the last frame, kept back for them, carries them
    assert _transcripts(frames) == [["Okay, "], ["look "], ["[this ", "over ", "here"]]


@pytest.mark.parametrize(
    "first",
    [
        # the alignment dropped the tag's first letters
        [_timed("ppy] ", 0.0, 0.0), _timed("Okay ", 0.1, 0.3)],
        # ... and the tail is glued to the next word, which keeps its timing
        [_timed("ppy]Okay ", 0.1, 0.3)],
        # ... and all of the tag but its closing bracket
        [_timed("] ", 0.0, 0.0), _timed("Okay ", 0.1, 0.3)],
    ],
)
async def test_emotion_frames_drops_the_tail_of_a_tag_whose_start_was_dropped(
    first: list[TimedString],
) -> None:
    node = _FakeNode([_timed_frame(300, *first), _timed_frame(400, _timed("then.", 0.3, 0.7))])
    frames = await _bracket_frames(node, "[happy] Okay then.")
    transcripts = [
        [(str(t), t.start_time) for t in f.userdata[USERDATA_TIMED_TRANSCRIPT]] for f in frames
    ]
    assert transcripts == [[("Okay ", 0.1)], [("then.", 0.3)]]
    assert _cue_times(frames) == [("happy", 0.0)]


async def test_emotion_frames_does_not_match_the_tail_of_a_dropped_tag_start() -> None:
    words = [
        # matched as a word, the tail would jump to the "excited" at the end
        ("xcited] ", 0.0, 0.0),
        ("We ", 0.0, 0.2),
        ("did ", 0.2, 0.4),
        ("it. ", 0.4, 0.6),
        ("[sad] ", 0.7, 0.7),
        ("Sadly ", 0.7, 1.1),
        ("I ", 1.1, 1.2),
        ("am ", 1.2, 1.4),
        ("still ", 1.4, 1.7),
        ("so ", 1.7, 1.9),
        ("excited.", 1.9, 2.5),
    ]
    node = _FakeNode([_timed_frame(2500, *(_timed(*w) for w in words))])
    frames = await _bracket_frames(node, "[excited] We did it. [sad] Sadly I am still so excited.")
    assert _cue_times(frames) == [("excited", 0.0), ("sad", 0.7)]


async def test_emotion_frames_skips_a_multi_word_tag_in_the_alignment() -> None:
    # the alignment splits at spaces, so the tag arrives as two tokens; matched as a word,
    # "[strong" would jump ahead to the "strong" at the end and place "sad" at 0.0
    node = _FakeNode(
        [
            _timed_frame(
                300,
                _timed("[strong ", 0.0, 0.0),
                _timed("accent] ", 0.0, 0.0),
                _timed("Okay. ", 0.0, 0.3),
            ),
            _timed_frame(
                400,
                _timed("[sad] ", 0.3, 0.3),
                _timed("That ", 0.4, 0.5),
                _timed("was ", 0.5, 0.6),
                _timed("strong.", 0.6, 0.7),
            ),
        ]
    )
    frames = await _bracket_frames(node, "[strong accent] Okay. [sad] That was strong.")
    assert _cue_times(frames) == [("sad", 0.4)]
    assert _transcripts(frames) == [["Okay. "], ["That ", "was ", "strong."]]


async def test_emotion_frames_keeps_a_later_word_that_ends_in_a_bracket() -> None:
    node = _FakeNode(
        [
            _timed_frame(300, _timed("[happy] ", 0.0, 0.0), _timed("Okay ", 0.0, 0.3)),
            _timed_frame(400, _timed("see] ", 0.3, 0.5), _timed("then.", 0.5, 0.7)),
        ]
    )
    frames = await _bracket_frames(node, "[happy] Okay see] then.")
    assert _transcripts(frames) == [["Okay "], ["see] ", "then."]]


async def test_emotion_frames_leaves_brackets_alone_in_the_expr_style() -> None:
    node = _FakeNode(
        [
            _timed_frame(300, _timed("ppy] ", 0.0, 0.0), _timed("See ", 0.0, 0.2)),
            _timed_frame(400, _timed("[1] ", 0.2, 0.5), _timed("there.", 0.5, 0.7)),
        ]
    )
    frames = [
        f
        async for f in _tts_node._emotion_frames(
            _chunks("ppy] See [1] there."), node, forward_expr=False, aligned=True
        )
    ]
    assert node.text == "ppy] See [1] there."
    assert _transcripts(frames) == [["ppy] ", "See "], ["[1] ", "there."]]


async def test_emotion_frames_closes_the_inner_node_early() -> None:
    node = _FakeNode([_frame(100) for _ in range(10)])
    gen = _tts_node._emotion_frames(
        _chunks("hi there friend"), node, forward_expr=False, aligned=False
    )
    await gen.__anext__()
    assert not node.closed
    await gen.aclose()
    assert node.closed


# ---------------------------------------------------------------------------
# emotion_tts_node in a running session
# ---------------------------------------------------------------------------

STRIPPED = " Hello there.  Now sad words."


class _SessionTTS(FakeTTS):
    """A ``FakeTTS`` with the given capabilities and markup dialect that records the text
    it is sent. Each synthesis (a stream, or a sentence through StreamAdapter) is 1 s."""

    def __init__(self, *, streaming: bool, aligned: bool = False, dialect: str = "") -> None:
        super().__init__(fake_audio_duration=1.0)
        self._capabilities = tts.TTSCapabilities(streaming=streaming, aligned_transcript=aligned)
        self.dialect = dialect
        self.text = ""

    class Markup(tts.TTS.Markup):
        def _provider_key(self) -> str:
            return cast(_SessionTTS, self._tts).dialect

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeChunkedStream:
        self.text += text
        return super().synthesize(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeSynthesizeStream:
        return _RecordingStream(tts=self, conn_options=conn_options)


class _RecordingStream(FakeSynthesizeStream):
    def push_text(self, token: str) -> None:
        cast(_SessionTTS, self._tts).text += token
        super().push_text(token)


async def _start_session(agent: Agent, session_tts: tts.TTS) -> AgentSession[None]:
    # expressive resolves on only for a TTS the framework can lower markup for
    session: AgentSession[None] = AgentSession(
        llm=FakeLLM(),
        tts=session_tts,
        vad=None,
        turn_handling={"turn_detection": None},
        expressive=True,
    )
    await session.start(agent)
    return session


async def _synthesize(
    session_tts: _SessionTTS, *chunks: str, **kwargs: Any
) -> list[rtc.AudioFrame]:
    """Run ``emotion_tts_node`` on ``chunks`` in a session using ``session_tts``."""
    agent = Agent(instructions="test")
    session = await _start_session(agent, session_tts)
    try:
        return [
            f async for f in emotion_tts_node(agent, _chunks(*chunks), ModelSettings(), **kwargs)
        ]
    finally:
        await session.aclose()


async def test_emotion_tts_node_lowers_markup_and_times_sentences_on_an_expressive_tts() -> None:
    # sentences under the adapter tokenizer's 20 characters merge into the next one
    sentences = (
        '<expr type="expression" label="happy"/> Hello there, it is good to see you. ',
        '<expr type="expression" label="sad"/> Now for some sad news.',
    )
    # non-streaming with a markup dialect: expressive resolves on and StreamAdapter wraps it
    session_tts = _SessionTTS(streaming=False, dialect="cartesia")
    frames = await _synthesize(session_tts, *sentences)

    assert session_tts.text == (
        '<emotion value="happy"/> Hello there, it is good to see you.'
        '<emotion value="sad"/> Now for some sad news.'
    )
    _, second = [t for f in frames for t in f.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])]
    assert _cue_times(frames) == [("happy", 0.0), ("sad", round(second.start_time, 3))]


async def test_emotion_tts_node_places_cues_when_promised_timings_never_come() -> None:
    session_tts = _SessionTTS(streaming=True, aligned=True)
    frames = await _synthesize(session_tts, *MARKED)

    assert session_tts.text == STRIPPED
    # placed by its share of the text, 10 of 21 characters, once the reply ends
    total = sum(f.duration for f in frames)
    assert _cue_times(frames) == [("happy", 0.0), ("sad", round(10 / 21 * total, 3))]


async def test_emotion_tts_node_without_timings_keeps_only_the_opening_marker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_tts = _SessionTTS(streaming=True)
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.keyframe"):
        frames = await _synthesize(session_tts, *MARKED)

    assert session_tts.text == STRIPPED, "the ignored marker is still stripped"
    assert _cue_times(frames) == [("happy", 0.0)]
    assert any("ignoring an emotion marker" in r.message for r in caplog.records)


async def test_emotion_tts_node_sends_only_the_allowed_emotions() -> None:
    frames = await _synthesize(
        _SessionTTS(streaming=True, aligned=True), *MARKED, emotions=("neutral", "sad")
    )
    assert [emotion for emotion, _ in _cue_times(frames)] == ["sad"]


@pytest.mark.parametrize(("max_cue_hold", "late"), [(0.2, True), (2.0, False)])
async def test_emotion_tts_node_holds_audio_back_for_a_pending_cue(
    max_cue_hold: float, late: bool
) -> None:
    # the promised word timings never come, so "sad" is placed only when the reply ends
    frames = await _synthesize(
        _SessionTTS(streaming=True, aligned=True), *MARKED, max_cue_hold=max_cue_hold
    )
    (sad,) = [
        cue
        for f in frames
        for cue in f.userdata.get(USERDATA_EMOTION_CUES, [])
        if cue.emotion == "sad"
    ]
    # a cue whose audio already went out rides a later frame, with a negative offset
    assert (sad.offset < 0) is late


@pytest.mark.parametrize(
    ("style", "aligned", "aligned_transcript", "warning"),
    [
        ("brackets", True, False, "set use_tts_aligned_transcript=True"),
        ("brackets", False, True, "needs word timings"),
        ("brackets", True, True, None),
        # square brackets are text in the expr style
        ("expr", True, False, None),
    ],
)
async def test_emotion_tts_node_warns_once_when_bracket_tags_reach_the_transcript(
    caplog: pytest.LogCaptureFixture,
    style: _tags.MarkerStyle,
    aligned: bool,
    aligned_transcript: bool,
    warning: str | None,
) -> None:
    agent = Agent(instructions="test", use_tts_aligned_transcript=aligned_transcript)
    session = await _start_session(agent, _SessionTTS(streaming=True, aligned=aligned))
    try:
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.keyframe"):
            for _ in range(2):
                node = emotion_tts_node(
                    agent, _chunks("[happy] Okay."), ModelSettings(), style=style
                )
                assert [f async for f in node]
    finally:
        await session.aclose()

    warnings = [r.message for r in caplog.records if "[tag] markers" in r.message]
    if warning is None:
        assert warnings == []
    else:
        assert len(warnings) == 1 and warning in warnings[0]


def test_emotion_instructions() -> None:
    expr = emotion_instructions()
    brackets = emotion_instructions(style="brackets")
    assert '<expr type="expression" label="..."/>' in expr
    assert all(e in expr for e in EMOTIONS)
    assert all(f"[{e}]" in brackets for e in EMOTIONS)
    # the expressive template is filled with str.format-style placeholders
    assert not re.search(r"[{}]", expr) and not re.search(r"[{}]", brackets)

    short = emotion_instructions(emotions=("neutral", "happy"))
    assert "happy" in short and "skeptical" not in short
    with pytest.raises(ValueError):
        emotion_instructions(style="other")  # type: ignore[arg-type]
