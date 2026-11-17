"""Deterministic speech providers and receiver observations shared by all room E2E tests."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import numpy as np

from livekit import rtc
from livekit.agents import LanguageCode, tts
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.tts import TTS, ChunkedStream, SpeechChunk, SynthesizeStream, TTSCapabilities
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

PHRASES = {500: "hello there agent", 700: "what time is it", 900: "early words before connect"}
AGENT_TONE = 1000


def loudness_and_pitch(pcm: bytes, sample_rate: int) -> tuple[float, int]:
    """RMS and the dominant frequency (to 100 Hz) of PCM16 mono."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if len(samples) < 32:
        return 0.0, 0
    peak = np.fft.rfftfreq(len(samples), 1 / sample_rate)[np.argmax(abs(np.fft.rfft(samples)))]
    return float(np.sqrt(np.mean(samples**2))), round(float(peak) / 100) * 100


def tone(frequency: int, seconds: float, sample_rate: int) -> bytes:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return (6000 * np.sin(2 * np.pi * frequency * t)).astype(np.int16).tobytes()


class ToneSTT(STT):
    """Hears a phrase's tone: interims grow a word per 150 ms, the final follows 300 ms of quiet."""

    def __init__(self, phrases: Mapping[int, str] = PHRASES) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=True))
        self.phrases = dict(phrases)

    async def _recognize_impl(self, buffer: object, **_: object) -> SpeechEvent:
        raise NotImplementedError

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS, **_: object
    ) -> RecognizeStream:
        return _ToneRecognition(stt=self, conn_options=conn_options, phrases=self.phrases)


class _ToneRecognition(RecognizeStream):
    def __init__(
        self, *, stt: ToneSTT, conn_options: APIConnectOptions, phrases: Mapping[int, str]
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self.phrases = phrases

    async def _run(self) -> None:
        phrase, heard, quiet, shown, pending = None, 0.0, 0.0, 0, b""
        async for frame in self.input:
            if not isinstance(frame, rtc.AudioFrame):
                continue
            rate, step = frame.sample_rate, frame.sample_rate // 50 * 2  # 20 ms, first channel
            pending += np.frombuffer(frame.data, np.int16)[:: frame.num_channels].tobytes()
            while len(pending) >= step:
                window, pending = pending[:step], pending[step:]
                rms, pitch = loudness_and_pitch(window, rate)
                if rms > 300 and pitch in self.phrases:
                    if phrase is None:
                        phrase, heard, shown = self.phrases[pitch], 0.0, 0
                        self._send(SpeechEventType.START_OF_SPEECH)
                    heard, quiet = heard + 0.02, 0.0
                    words = phrase.split()[: 1 + int(heard / 0.15)]
                    if len(words) > shown:
                        shown = len(words)
                        self._send(SpeechEventType.INTERIM_TRANSCRIPT, " ".join(words))
                elif phrase is not None:
                    quiet += 0.02
                    if quiet >= 0.3:
                        self._send(SpeechEventType.FINAL_TRANSCRIPT, phrase)
                        self._send(SpeechEventType.END_OF_SPEECH)
                        phrase = None

    def _send(self, kind: SpeechEventType, text: str = "") -> None:
        said = [SpeechData(text=text, language=LanguageCode("en"))] if text else []
        self.send(SpeechEvent(type=kind, alternatives=said))


class ToneTTS(TTS):
    """Speaks a 1 kHz tone, 60 ms per character, accepting Cartesia's markup."""

    class Markup(TTS.Markup):
        def _provider_key(self) -> str:
            return "cartesia"

    def __init__(self) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=True), sample_rate=24000, num_channels=1
        )

    def _synthesize(self, chunk: SpeechChunk, *, conn_options: APIConnectOptions) -> ChunkedStream:
        return _ToneChunks(tts=self, input_text=chunk.said, conn_options=conn_options)

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        return _ToneStream(tts=self, conn_options=conn_options)


def _speak(emitter: tts.AudioEmitter, text: str) -> None:
    audio = tone(AGENT_TONE, min(6.0, 0.06 * max(len(text), 5)), 24000)
    for offset in range(0, len(audio), 960):
        emitter.push(audio[offset : offset + 960])


class _ToneChunks(ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id="tone", sample_rate=24000, num_channels=1, mime_type="audio/pcm"
        )
        _speak(output_emitter, self._input_text)
        output_emitter.flush()


class _ToneStream(SynthesizeStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id="tone", sample_rate=24000, num_channels=1, mime_type="audio/pcm", stream=True
        )
        text = ""
        async for data in self.input:
            if isinstance(data, SpeechChunk):
                text += data.said
            elif text.strip():
                output_emitter.start_segment(segment_id=f"tone-{id(text)}")
                _speak(output_emitter, text)
                output_emitter.flush()
                text = ""


Transcription = tuple[str, dict[str, str], rtc.TextStreamReader]  # sender, header, stream


def read_transcriptions(room: rtc.Room) -> asyncio.Queue[Transcription]:
    """Every ``lk.transcription`` stream as it opens, as components-js ``useTranscriptions``
    receives them: the sender, the header's attributes, and the stream (whose
    ``info.attributes`` take the trailer's once read to its end)."""
    opened: asyncio.Queue[Transcription] = asyncio.Queue()
    room.register_text_stream_handler(
        "lk.transcription",
        lambda reader, sender: opened.put_nowait(
            (sender, dict(reader.info.attributes or {}), reader)
        ),
    )
    return opened


async def next_transcription(
    transcriptions: asyncio.Queue[Transcription], sender: str, timeout: float = 15
) -> tuple[dict[str, str], rtc.TextStreamReader]:
    """The next stream ``sender`` opens."""
    async with asyncio.timeout(timeout):
        while (transcription := await transcriptions.get())[0] != sender:
            pass
    return transcription[1], transcription[2]


async def publish_microphone(
    room: rtc.Room, *, preconnect: bool = False, sample_rate: int = 48000
) -> tuple[rtc.AudioSource, rtc.LocalTrackPublication]:
    source = rtc.AudioSource(sample_rate, 1)
    options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_MICROPHONE, preconnect_buffer=preconnect
    )
    track = rtc.LocalAudioTrack.create_audio_track("microphone", source)
    return source, await room.local_participant.publish_track(track, options)


async def speak(
    source: rtc.AudioSource,
    phrase: str,
    seconds: float = 0.8,
    *,
    phrases: Mapping[int, str] = PHRASES,
) -> None:
    """Say ``phrase`` into the microphone in real time, then stay quiet for 0.6 s."""
    pitch = next(f for f, text in phrases.items() if text == phrase)
    await play(source, tone(pitch, seconds, 48000) + bytes(48000 * 2 * 6 // 10))


async def play(source: rtc.AudioSource, pcm: bytes) -> None:
    for offset in range(0, len(pcm), 960):
        chunk = pcm[offset : offset + 960].ljust(960, b"\0")
        await source.capture_frame(
            rtc.AudioFrame(data=chunk, sample_rate=48000, num_channels=1, samples_per_channel=480)
        )


async def track_of(
    participant: rtc.RemoteParticipant, source: rtc.TrackSource.ValueType
) -> rtc.Track:
    """``participant``'s subscribed track of ``source``, once it publishes one (clients pick the
    avatar's audio and video by source: components-js ``useAgent.ts:179-202``)."""
    async with asyncio.timeout(20):
        while not (
            tracks := [
                p.track
                for p in participant.track_publications.values()
                if p.source == source and p.track is not None
            ]
        ):
            await asyncio.sleep(0.01)
    return tracks[0]


class Heard:
    """What the person hears on a participant's microphone: when each 10 ms frame carried the
    agent's tone."""

    def __init__(self) -> None:
        self.tone_at: list[float] = []

    @property
    def seconds(self) -> float:
        return len(self.tone_at) / 100


class Seen:
    """What the person sees on a participant's camera: how many frames were lit (the face
    mouthing speech) and dark (idle)."""

    def __init__(self) -> None:
        self.lit = 0
        self.dark = 0


@asynccontextmanager
async def hearing(participant: rtc.RemoteParticipant) -> AsyncIterator[Heard]:
    heard = Heard()

    async def listen() -> None:
        track = await track_of(participant, rtc.TrackSource.SOURCE_MICROPHONE)
        stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1, frame_size_ms=10)
        try:
            async for event in stream:
                rms, pitch = loudness_and_pitch(bytes(event.frame.data), 48000)
                if rms > 300 and pitch == AGENT_TONE:
                    heard.tone_at.append(time.monotonic())
        finally:
            await stream.aclose()

    task = asyncio.create_task(listen())
    try:
        yield heard
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@asynccontextmanager
async def watching(participant: rtc.RemoteParticipant) -> AsyncIterator[Seen]:
    seen = Seen()

    async def watch() -> None:
        track = await track_of(participant, rtc.TrackSource.SOURCE_CAMERA)
        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGBA)
        try:
            async for event in stream:
                brightness = np.frombuffer(bytes(event.frame.data), dtype=np.uint8)[0::4].mean()
                if brightness > 192:
                    seen.lit += 1
                elif brightness < 64:
                    seen.dark += 1
        finally:
            await stream.aclose()

    task = asyncio.create_task(watch())
    try:
        yield seen
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
