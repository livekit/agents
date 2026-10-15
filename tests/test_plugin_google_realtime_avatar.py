from __future__ import annotations

import asyncio
import io
import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fractions import Fraction

import av
import numpy as np
import pytest
from google.genai import types

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.plugins.google.realtime import AvatarSession
from livekit.plugins.google.realtime._avatar_stream import AvatarStreamDecoder
from livekit.plugins.google.realtime.avatar import _AvatarPlayoutOutput
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit

_WIDTH, _HEIGHT, _FPS, _SAMPLE_RATE = 64, 96, 24, 24000


def _fragmented_mp4_chunks(seconds: float = 2.0) -> list[bytes]:
    """An H.264 + AAC stream cut the way Gemini Live sends it: the init segment, then one
    `moof`/`mdat` pair per chunk."""
    out = io.BytesIO()
    container = av.open(
        out,
        mode="w",
        format="mp4",
        options={"movflags": "frag_keyframe+empty_moov+default_base_moof+frag_every_frame"},
    )
    video = container.add_stream("libx264", rate=_FPS)
    video.width, video.height, video.pix_fmt = _WIDTH, _HEIGHT, "yuv420p"
    audio = container.add_stream("aac", rate=_SAMPLE_RATE)
    audio.layout = "mono"

    tone = (np.sin(np.arange(_SAMPLE_RATE // _FPS) / 4) * 8000).astype(np.int16)
    for i in range(int(seconds * _FPS)):
        image = np.full((_HEIGHT, _WIDTH, 3), i * 8 % 256, dtype=np.uint8)
        for packet in video.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
        samples = av.AudioFrame.from_ndarray(tone[None, :], format="s16", layout="mono")
        samples.sample_rate = _SAMPLE_RATE
        samples.pts = i * len(tone)
        samples.time_base = Fraction(1, _SAMPLE_RATE)
        for packet in audio.encode(samples):
            container.mux(packet)
    for stream in (video, audio):
        for packet in stream.encode():
            container.mux(packet)
    container.close()

    data = out.getvalue()
    chunks: list[bytes] = []
    start = offset = 0
    while offset < len(data):
        size, box = struct.unpack(">I4s", data[offset : offset + 8])
        offset += size
        if box in (b"moov", b"mdat"):
            chunks.append(data[start:offset])
            start = offset
    return chunks


def _video_message(chunk: bytes) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            model_turn=types.Content(
                parts=[types.Part(inline_data=types.Blob(data=chunk, mime_type="video/mp4"))]
            )
        )
    )


class _FakeAvatar:
    def __init__(self) -> None:
        self.frames: list[rtc.VideoFrame | rtc.AudioFrame] = []

    def _push_frame(self, frame: rtc.VideoFrame | rtc.AudioFrame) -> None:
        self.frames.append(frame)


@asynccontextmanager
async def _avatar_session(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[RealtimeSession]:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    model = RealtimeModel(
        model="gemini-3.8-live", avatar_config=types.AvatarConfig(avatar_name="Kira")
    )
    session = model.session()
    session._msg_ch.close()
    await utils.aio.cancel_and_wait(session._main_atask)
    session._avatar_decoder = AvatarStreamDecoder(
        on_video=session._on_avatar_video, on_audio=session._on_avatar_audio
    )
    try:
        yield session
    finally:
        await session.aclose()


async def _wait_for(condition: object, timeout: float = 5.0) -> None:
    async def _poll() -> None:
        while not condition():  # type: ignore[operator]
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout)


async def test_avatar_config_reaches_the_connect_config(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _avatar_session(monkeypatch) as session:
        config = session._build_connect_config()

        assert config.avatar_config == types.AvatarConfig(avatar_name="Kira")
        assert config.response_modalities == [types.Modality.VIDEO]
        assert session._gemini_model.capabilities.audio_output


def test_avatar_config_requires_the_video_modality(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    with pytest.raises(ValueError):
        RealtimeModel(
            avatar_config=types.AvatarConfig(avatar_name="Kira"),
            modalities=[types.Modality.AUDIO],
        )
    with pytest.raises(ValueError):
        RealtimeModel(modalities=[types.Modality.VIDEO])


async def test_idle_avatar_stream_reaches_the_avatar_without_a_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server streams idle frames while the model listens. They must be published, and
    must not open a generation or be played as PCM."""
    async with _avatar_session(monkeypatch) as session:
        avatar = _FakeAvatar()
        session._gemini_model._avatar_session = avatar  # type: ignore[assignment]
        created: list[object] = []
        session.on("generation_created", created.append)

        for chunk in _fragmented_mp4_chunks():
            assert not session._route_avatar_media(_video_message(chunk))

        await _wait_for(lambda: sum(isinstance(f, rtc.VideoFrame) for f in avatar.frames) >= 8)
        video = [f for f in avatar.frames if isinstance(f, rtc.VideoFrame)]
        audio = [f for f in avatar.frames if isinstance(f, rtc.AudioFrame)]
        assert (video[0].width, video[0].height) == (_WIDTH, _HEIGHT)
        assert audio and all(f.sample_rate == _SAMPLE_RATE for f in audio)
        assert session._current_generation is None
        assert not created


async def test_avatar_speech_feeds_the_generation_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _avatar_session(monkeypatch) as session:
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        message = _video_message(b"")
        assert message.server_content is not None and message.server_content.model_turn
        message.server_content.output_transcription = types.Transcription(text="Hello")
        message.server_content.model_turn.parts = [
            types.Part(inline_data=types.Blob(data=c, mime_type="video/mp4"))
            for c in _fragmented_mp4_chunks()
        ]
        assert session._route_avatar_media(message)
        assert message.server_content.model_turn is None
        session._handle_server_content(message.server_content)

        await _wait_for(lambda: not gen.audio_ch.empty())
        frame = gen.audio_ch.recv_nowait()
        assert frame.sample_rate == _SAMPLE_RATE
        assert gen.output_text == "Hello"


async def test_interrupted_reply_is_silenced_until_the_next_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    model = RealtimeModel(avatar_config=types.AvatarConfig(avatar_name="Kira"))
    avatar = AvatarSession(model)
    output = _AvatarPlayoutOutput(avatar)
    finished: list[PlaybackFinishedEvent] = []
    output.on("playback_finished", finished.append)
    speech = rtc.AudioFrame(
        data=b"\x10\x00" * 240, sample_rate=_SAMPLE_RATE, num_channels=1, samples_per_channel=240
    )

    await output.capture_frame(speech)
    output.clear_buffer()
    avatar._push_frame(speech)

    assert finished and finished[0].interrupted
    muted, _ = avatar._frame_ch.recv_nowait()
    assert isinstance(muted, rtc.AudioFrame) and not any(bytes(muted.data))

    output.flush()
    await output.capture_frame(speech)
    avatar._push_frame(speech)
    unmuted, _ = avatar._frame_ch.recv_nowait()
    assert isinstance(unmuted, rtc.AudioFrame) and bytes(unmuted.data) == bytes(speech.data)


async def test_avatar_is_silent_while_the_session_audio_output_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    model = RealtimeModel(avatar_config=types.AvatarConfig(avatar_name="Kira"))
    avatar = AvatarSession(model)
    output = _AvatarPlayoutOutput(avatar)
    speech = rtc.AudioFrame(
        data=b"\x10\x00" * 240, sample_rate=_SAMPLE_RATE, num_channels=1, samples_per_channel=240
    )

    output.on_detached()
    avatar._push_frame(speech)
    silenced, _ = avatar._frame_ch.recv_nowait()
    assert isinstance(silenced, rtc.AudioFrame) and not any(bytes(silenced.data))

    output.on_attached()
    avatar._push_frame(speech)
    restored, _ = avatar._frame_ch.recv_nowait()
    assert isinstance(restored, rtc.AudioFrame) and bytes(restored.data) == bytes(speech.data)


def _solid_frame(value: int) -> rtc.VideoFrame:
    return rtc.VideoFrame(
        _WIDTH, _HEIGHT, rtc.VideoBufferType.RGBA, bytes([value]) * (_WIDTH * _HEIGHT * 4)
    )


def _audio(peak: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=np.full(240, peak, dtype=np.int16).tobytes(),
        sample_rate=_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=240,
    )


def _published_video(avatar: AvatarSession) -> list[int]:
    values: list[int] = []
    while not avatar._frame_ch.empty():
        frame, _ = avatar._frame_ch.recv_nowait()
        if isinstance(frame, rtc.VideoFrame):
            values.append(bytes(frame.data)[0])
    return values


async def test_silenced_avatar_loops_listening_frames_instead_of_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While the session audio is off, the server can keep rendering the reply, even after
    the turn completed. Video that comes with speech audio is replaced by a loop of frames
    rendered while the avatar was quiet."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    model = RealtimeModel(avatar_config=types.AvatarConfig(avatar_name="Kira"))
    avatar = AvatarSession(model)
    output = _AvatarPlayoutOutput(avatar)

    for value in range(1, 10):
        avatar._push_frame(_audio(0))
        avatar._push_frame(_solid_frame(value))
    assert _published_video(avatar) == list(range(1, 10))

    output.on_detached()
    for _ in range(4):
        avatar._push_frame(_audio(8000))
        avatar._push_frame(_solid_frame(200))
    assert _published_video(avatar) == [1, 2, 3, 2]

    output.on_attached()
    avatar._push_frame(_audio(8000))
    avatar._push_frame(_solid_frame(200))
    assert _published_video(avatar) == [200]


async def test_closing_the_avatar_leaves_the_agent_in_the_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The avatar is published by the agent participant, so closing it must not remove
    that participant from the room."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    model = RealtimeModel(avatar_config=types.AvatarConfig(avatar_name="Kira"))
    avatar = AvatarSession(model)
    removed: list[object] = []

    class _Room:
        name = "room"
        local_participant = type("P", (), {"identity": "agent"})()

        def isconnected(self) -> bool:
            return True

        def off(self, *_: object) -> None:
            pass

    class _JobContext:
        class api:
            class room:
                @staticmethod
                async def remove_participant(request: object) -> None:
                    removed.append(request)

    avatar._room = _Room()  # type: ignore[assignment]
    monkeypatch.setattr(
        "livekit.agents.voice.avatar._types.get_job_context", lambda required=False: _JobContext()
    )
    await avatar.aclose()
    assert not removed


async def test_reply_audio_closes_only_after_its_fragments_are_decoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generation_complete can arrive while the reply's last fragments are still decoding.
    The generation's audio must stay open until they are delivered."""
    async with _avatar_session(monkeypatch) as session:
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None
        chunks = _fragmented_mp4_chunks()
        for chunk in chunks:
            assert not session._route_avatar_media(_video_message(chunk))
        session._handle_server_content(types.LiveServerContent(generation_complete=True))

        await _wait_for(lambda: gen.audio_ch.closed)
        received = 0.0
        while not gen.audio_ch.empty():
            received += gen.audio_ch.recv_nowait().duration
        assert received >= 1.5
