import asyncio
from collections import deque
import contextlib
from typing import Literal
from attrs import evolve, define

from livekit import rtc

from . import aio
from . import llm as allm
from . import stt as astt
from . import tts as atts
from . import vad as avad


@define
class _PlayoutSync:
    type = Literal[
        "synthesize",  # for TTS
        "clear",  # clear playout buffer
        "validate",  # valdiate buffer, start playout
    ]
    text: str | None = None


@define
class _VADSync:
    type = Literal["user_interruption",]


class ConvAgent:
    def __init__(
        self,
        *,
        vad: avad.VAD,
        tts: atts.TTS,
        llm: allm.LLM,
        stt: astt.STT,
        fnc_ctx: allm.FunctionContext | None = None,
        allow_interruptions: bool = True,
        # TODO(theomonnom): document how this is dependent on the given VAD parameters the user has set (default values are OK for now)
        # or mb substracts the VAD parameters from the interrupt_speech_duration?
        interrupt_speech_duration: float = 0.5,
        interrupt_speech_validation: float = 1.5,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        loop = loop or asyncio.get_event_loop()
        self._vad, self._tts, self._llm, self._stt = vad, tts, llm, stt
        self._fnc_ctx = fnc_ctx
        self._allow_interruptions = allow_interruptions
        self._int_speech_duration = interrupt_speech_duration
        self._int_speech_validation = interrupt_speech_validation

        self._chat_ctx = allm.ChatContext()
        self._llm_gen: asyncio.Task | None = None

        self._int_start_sleep, self._int_validate_sleep = None, None

    async def run(
        self, audio_stream: rtc.AudioStream, audio_source: rtc.AudioSource
    ) -> None:
        def _interp(a: float, b: float, t: float) -> float:
            return a + (b - a) * (3 * t**2 - 2 * t**3)

        vad_stream = self._vad.stream()
        stt_stream = self._stt.stream()
        tts_stream = self._tts.stream()

        llm_task: asyncio.Task | None = None

        po_buffer = asyncio.Queue[rtc.AudioFrame]()
        po_validate = asyncio.Event()  # decides when to start playout
        po_task = None

        user_speech = ""
        user_speaking = False

        async def _playout_task():
            """
            Playout task that consumes the playout buffer and sends it to the audio source
            Also apply volume reduction when the user is speaking
            """
            while True:
                frame = await buffer.get()
                await asyncio.shield(audio_source.capture_frame(frame))

        async def _llm_task(self, speech: str) -> None:
            if self._llm_gen is not None:
                self._llm_gen.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._llm_gen

            chat_ctx = evolve(self._chat_ctx)
            chat_ctx.messages.append(
                allm.ChatMessage(role=allm.ChatRole.USER, text=speech)
            )

            stream: allm.LLMStream | None = None
            try:
                stream = await self._llm.chat(chat_ctx, self._fnc_ctx)
                async for chunk in stream:
                    # forward the chunk to the playout task
                    pass
            except asyncio.CancelledError:
                if stream is not None:
                    # wait=False will cancel running functions
                    await asyncio.shield(stream.aclose(wait=False))

        select = aio.select([vad_stream, tts_stream, stt_stream, audio_stream])
        async with contextlib.aclosing(select) as select:
            async for s in select:
                if s.selected is audio_stream:
                    # forward user audio to VAD and STT
                    audio_event: rtc.AudioFrameEvent = s.result()
                    vad_stream.push_frame(audio_event.frame)
                    stt_stream.push_frame(audio_event.frame)

                elif s.selected is vad_stream:
                    # handle VAD events
                    vad_event: avad.VADEvent = s.result()
                    if vad_event.type == avad.VADEventType.START_OF_SPEECH:
                        user_speaking = True
                    elif vad_event.type == avad.VADEventType.END_OF_SPEECH:
                        user_speaking = False

                elif s.selected is stt_stream:
                    # handle STT events
                    stt_event: astt.SpeechEvent = s.result()
                    if stt_event.type == astt.SpeechEventType.FINAL_TRANSCRIPT:
                        alt = stt_event.alternatives[0]
                        user_speech += f"{alt.text} "

                        if not user_speaking:
                            # cancel everything and start the LLM call
                            await self._make_llm_call(self._user_speech)

                    elif stt_event.type == astt.SpeechEventType.END_OF_SPEECH:
                        self._user_speech = ""
                        po_validate.set()

                elif s.selected is tts_stream:
                    # add TTS audio to the playout buffer
                    tts_event: atts.SynthesisEvent = s.result()
                    if tts_event.type == atts.SynthesisEventType.AUDIO:
                        assert tts_event.audio
                        po_buffer.put_nowait(tts_event.audio.data)

                elif s.selected is self._playout_ch:
                    res: _PlayoutSync = s.result()
                    if res.type == "clear":
                        buffer = asyncio.Queue()
                        # recreate a new tts stream
                        await tts_stream.aclose(wait=False)
                        tts_stream = self._tts.stream()
                    elif res.type == "synthesize":
                        tts_stream.push_text(res.text)
                    elif res.type == "validate":
                        playout_task = asyncio.create_task(_playout())


"""
            if self._llm_gen is not None:  # TODO Also TTS
                self._int_start_sleep = aio.sleep(self._int_speech_duration)
                self._int_validate_sleep = aio.sleep(
                    self._int_speech_duration + self._int_speech_validation
                )

                async def _interrupt_validator_h():
                    assert self._int_start_sleep
                    assert self._int_validate_sleep
                    await self._int_start_sleep
                    # start to decrease the volume here?

                    await self._int_validate_sleep

                    # interruption validation passed, interrupt the LLM/FNC/TTS
                    await self._handle_interruption()

                self._int_validator_task = asyncio.create_task(_interrupt_validator_h())

                """
