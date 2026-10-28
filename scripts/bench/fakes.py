"""Paced provider fakes for the same generated provider crossings as smoke.py.

100 ms STT finalization, 30 ms LLM chunks, 80 ms TTS startup, 150 words/minute;
VAD reports every 32 ms. These are declared workload assumptions, not model CPU claims.
"""

from __future__ import annotations

import asyncio
from typing import Never

from livekit import rtc
from livekit.agents import (
    VAD,
    APIConnectOptions,
    LanguageCode,
    VADCapabilities,
    VADEvent,
    VADEventType,
    VADStream,
    llm,
    stt,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS


class PacedLLM(llm.LLM[Never]):
    def chat(
        self, *, chat_ctx: llm.ChatContext, tools: list[llm.Tool] | None = None, **_: object
    ) -> llm.LLMStream:
        return Reply(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=DEFAULT_API_CONNECT_OPTIONS
        )


class Reply(llm.LLMStream):
    async def _run(self) -> None:
        for word in ("I can help. ", "Your request is ready."):
            await asyncio.sleep(0.03)
            self.send(
                llm.ChatChunk(id="bench", delta=llm.ChoiceDelta(role="assistant", content=word))
            )


class PacedSTT(stt.STT[Never]):
    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))
        self.frames = 0
        self.pcm_bytes = 0
        self.finals = 0

    async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
        raise NotImplementedError

    def stream(self, **_: object) -> stt.RecognizeStream:
        return Recognize(self)


class Recognize(stt.RecognizeStream):
    def __init__(self, owner: PacedSTT) -> None:
        super().__init__(stt=owner, conn_options=DEFAULT_API_CONNECT_OPTIONS)
        self.owner = owner

    async def _run(self) -> None:
        heard = 0.0
        async for frame in self.input:
            if not isinstance(frame, rtc.AudioFrame):
                continue
            self.owner.frames += 1
            self.owner.pcm_bytes += frame.data.nbytes
            heard += frame.duration
            if heard >= 0.4 and not self.owner.finals:
                self.send(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
                await asyncio.sleep(0.1)
                self.send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                language=LanguageCode("en"), text="Please help me.", confidence=1
                            )
                        ],
                    )
                )
                self.send(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
                self.owner.finals += 1


class PacedVAD(VAD[Never]):
    def __init__(self) -> None:
        super().__init__(capabilities=VADCapabilities(update_interval=0.032))
        self.frames = 0
        self.pcm_bytes = 0
        self.predictions = 0

    def stream(self) -> VADStream:
        return Detect(self)


class Detect(VADStream):
    def __init__(self, owner: PacedVAD) -> None:
        super().__init__(owner)
        self.owner = owner

    async def _main_task(self) -> None:
        duration = 0.0
        reported = 0.0
        started = False
        ended = False
        async for frame in self.input:
            if not isinstance(frame, rtc.AudioFrame):
                continue
            self.owner.frames += 1
            self.owner.pcm_bytes += frame.data.nbytes
            duration += frame.duration
            while duration - reported >= 0.032:
                reported += 0.032
                self.owner.predictions += 1
                self.send(
                    VADEvent(
                        type=VADEventType.INFERENCE_DONE,
                        samples_index=int(duration * 16000),
                        timestamp=duration,
                        speech_duration=min(duration, 0.4),
                        silence_duration=max(0, duration - 0.4),
                        inference_duration=0,
                        probability=1 if duration < 0.4 else 0,
                    )
                )
            kind = None
            if not started:
                started = True
                kind = VADEventType.START_OF_SPEECH
            elif duration >= 0.4 and not ended:
                ended = True
                kind = VADEventType.END_OF_SPEECH
            if kind is not None:
                self.send(
                    VADEvent(
                        type=kind,
                        samples_index=int(duration * 16000),
                        timestamp=duration,
                        speech_duration=min(duration, 0.4),
                        silence_duration=0,
                    )
                )


class PacedTTS(tts.TTS[Never]):
    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1
        )
        self.frames = 0
        self.started = 0
        self.completed = 0

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return Synthesize(self, text, conn_options)


class Synthesize(tts.ChunkedStream):
    def __init__(self, owner: PacedTTS, text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=owner, input_text=text, conn_options=conn_options)
        self.owner = owner

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        self.owner.started += 1
        await asyncio.sleep(0.08)
        output_emitter.initialize(
            request_id="bench", sample_rate=24000, num_channels=1, mime_type="audio/pcm"
        )
        for _ in range(max(1, len(self.input_text.split())) * 20):
            output_emitter.push(b"\0\0" * 480)
            self.owner.frames += 1
            await asyncio.sleep(0.02)
        output_emitter.flush()
        self.owner.completed += 1
