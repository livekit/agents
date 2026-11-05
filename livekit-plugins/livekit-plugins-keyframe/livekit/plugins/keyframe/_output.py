from __future__ import annotations

import asyncio

import numpy as np

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.agents.voice.avatar._datastream_io import AUDIO_STREAM_TOPIC
from livekit.agents.voice.io import AudioOutput

from ._framing import (
    EMOTION_NOW_PROTOCOL,
    FRAMING_ATTRIBUTE,
    FRAMING_VERSION,
    PROTOCOL_ATTRIBUTE,
    encode_audio,
    encode_cue,
)
from .log import logger
from .types import USERDATA_EMOTION_CUES, Emotion, EmotionCue

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
CONTROL_TOPIC = "lk.control"


class KeyframeAudioOutput(DataStreamAudioOutput):
    """Streams agent audio to the Keyframe avatar, with timed emotion cues.

    With ``framed`` (a ``persona-2.0-live`` persona) each segment is a framed
    ``lk.audio_stream``: audio records interleaved with cue records placed on the
    stream's sample clock. The cues are the :data:`USERDATA_EMOTION_CUES` on the frames
    the ``tts_node`` yields, timed against the frame's audio (``emotion_tts_node``
    attaches these). Audio is resampled to 24 kHz mono here rather than by the
    framework, which would drop the frame ``userdata`` carrying the cues. Without
    ``framed`` (``persona-1.5-live``) segments are raw PCM and cues on frames are ignored.

    :meth:`set_emotion` is independent of the audio stream: an ``lk.control`` message the
    avatar applies from the next video frame it renders.
    """

    def __init__(self, room: rtc.Room, *, destination_identity: str, framed: bool) -> None:
        super().__init__(
            room,
            destination_identity=destination_identity,
            sample_rate=None,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
            clear_buffer_timeout=None,
            wait_playback_start=True,
        )
        self._framed = framed
        self._warned_ignored_cues = False
        self._write_lock = asyncio.Lock()
        self._resampler: rtc.AudioResampler | None = None
        self._resampler_format: tuple[int, int] | None = None
        # seconds of input audio captured for the open segment, before resampling
        self._input_duration = 0.0

    # -- audio path ----------------------------------------------------------

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if self._start_atask is None:
            self._start_atask = asyncio.create_task(self._start_task())

        await asyncio.shield(self._start_atask)

        # skip DataStreamAudioOutput.capture_frame: it would write the raw frame
        await AudioOutput.capture_frame(self, frame)

        cues: list[EmotionCue] = list(frame.userdata.get(USERDATA_EMOTION_CUES) or ())
        if cues and not self._framed:
            if not self._warned_ignored_cues:
                self._warned_ignored_cues = True
                logger.warning(
                    "ignoring emotion cues: timed emotions need a persona-2.0-live persona",
                    extra={"lk.pii.participant_identity": self._destination_identity},
                )
            cues = []

        async with self._write_lock:
            if self._stream_writer is None:
                await self._open_stream()
            writer = self._stream_writer
            assert writer is not None

            frame_start = self._input_duration
            self._input_duration += frame.duration

            frames = self._resample(frame)
            pcm = b"".join(bytes(f.data) for f in frames)
            self._pushed_duration += sum(f.duration for f in frames)

            if self._framed:
                records = [
                    encode_cue(cue.emotion, max(0, round((frame_start + cue.offset) * SAMPLE_RATE)))
                    for cue in cues
                ]
                if pcm:
                    records.append(encode_audio(pcm))
                if records:
                    await writer.write(b"".join(records))
                return

            if pcm:
                await writer.write(pcm)

    async def _open_stream(self) -> None:
        attributes = {"sample_rate": str(SAMPLE_RATE), "num_channels": str(NUM_CHANNELS)}
        if self._framed:
            attributes[FRAMING_ATTRIBUTE] = FRAMING_VERSION

        self._stream_writer = await self._room.local_participant.stream_bytes(
            name=utils.shortuuid("AUDIO_"),
            topic=AUDIO_STREAM_TOPIC,
            destination_identities=[self._destination_identity],
            attributes=attributes,
        )
        self._pushed_duration = 0.0
        self._input_duration = 0.0

    def _resample(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        if frame.num_channels != NUM_CHANNELS:
            frame = _downmix(frame)

        fmt = (frame.sample_rate, frame.num_channels)
        out: list[rtc.AudioFrame] = []
        if self._resampler is not None and self._resampler_format != fmt:
            # the input format changed mid-segment: the old resampler's tail goes out
            # first, or the stream would lose it and run ahead of the cue offsets
            out.extend(self._resampler.flush())
            self._resampler = None
            self._resampler_format = None

        if frame.sample_rate == SAMPLE_RATE:
            out.append(frame)
            return out

        if self._resampler is None:
            self._resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )
            self._resampler_format = fmt
        out.extend(self._resampler.push(frame))
        return out

    def flush(self) -> None:
        AudioOutput.flush(self)
        if self._stream_writer is None or not self._started:
            return

        writer, resampler = self._stream_writer, self._resampler
        self._stream_writer = None
        self._resampler = None
        self._resampler_format = None

        task = asyncio.create_task(self._close_stream(writer, resampler))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _close_stream(
        self, writer: rtc.ByteStreamWriter, resampler: rtc.AudioResampler | None
    ) -> None:
        async with self._write_lock:
            if resampler is not None:
                tail = b"".join(bytes(f.data) for f in resampler.flush())
                if tail:
                    await writer.write(encode_audio(tail) if self._framed else tail)
            await writer.aclose()

    # -- emotion -------------------------------------------------------------

    async def set_emotion(self, emotion: Emotion) -> None:
        """Change the avatar's expression from the next video frame it renders.

        An avatar advertising protocol 2 or higher gets ``emotion_now:``, which switches
        regardless of the audio it has received. An older avatar, or one that has not
        advertised a protocol yet, gets ``emotion:``, which it applies after the audio
        received so far (that message keeps its meaning for clients that interleave
        emotion changes with their audio writes).
        """
        kind = "emotion_now" if self._avatar_protocol() >= EMOTION_NOW_PROTOCOL else "emotion"
        await self._room.local_participant.publish_data(
            f"{kind}:{emotion}".encode(),
            reliable=True,
            topic=CONTROL_TOPIC,
            destination_identities=[self._destination_identity],
        )

    def _avatar_protocol(self) -> int:
        """The protocol version the avatar advertises; 0 until it has."""
        participant = self._room.remote_participants.get(self._destination_identity)
        attributes = getattr(participant, "attributes", None) or {}
        value = attributes.get(PROTOCOL_ATTRIBUTE, "")
        return int(value) if value.isdigit() else 0


def _downmix(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
    mono = samples.astype(np.int32).mean(axis=1).astype(np.int16)
    return rtc.AudioFrame(
        data=mono.tobytes(),
        sample_rate=frame.sample_rate,
        num_channels=NUM_CHANNELS,
        samples_per_channel=frame.samples_per_channel,
    )
