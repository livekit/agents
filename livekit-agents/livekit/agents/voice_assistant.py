import asyncio
import contextlib

from attrs import evolve
from livekit import rtc

from . import aio
from . import llm as allm
from . import stt as astt
from . import tts as atts
from . import vad as avad
from .log import logger


class VoiceAssistant:
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
        base_volume: float = 1.0,
        interrupt_speech_duration: float = 0.5,
        interrupt_speech_validation: float = 1.5,
        interrupt_volume: float = 0.3,
        interrupt_volume_duration: float = 1,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        loop = loop or asyncio.get_event_loop()
        self._vad, self._tts, self._llm, self._stt = vad, tts, llm, stt
        self._fnc_ctx = fnc_ctx
        self._allow_interruptions = allow_interruptions
        self._base_volume = base_volume
        self._int_speech_duration = interrupt_speech_duration
        self._int_speech_validation = interrupt_speech_validation
        self._int_volume = interrupt_volume
        self._int_volume_duration = interrupt_volume_duration
        self._chat_ctx = allm.ChatContext()

    async def run(
        self, audio_stream: rtc.AudioStream, audio_source: rtc.AudioSource
    ) -> None:
        vad_stream = self._vad.stream()
        stt_stream = self._stt.stream()

        po_validate = asyncio.Condition()

        play_task: asyncio.Task | None = None
        synth_task: asyncio.Task | None = None
        int_tx: aio.ChanSender[bool] | None = None
        int_task: asyncio.Task | None = None

        user_speech = ""
        agent_speaking = False

        async def _cancel_synthesis():
            logger.debug("_cancel_synthesis")
            nonlocal synth_task, play_task, int_tx
            if synth_task is not None:
                assert play_task
                synth_task.cancel()
                play_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.gather(synth_task, play_task, return_exceptions=True)

                synth_task = play_task = int_tx = None

        async def _start_synthesis(text: str) -> None:
            logger.debug("_start_synthesis")
            nonlocal synth_task, play_task, int_tx
            await _cancel_synthesis()

            async def _playout_task(
                chat_ctx: allm.ChatContext,
                po_rx: aio.ChanReceiver[rtc.AudioFrame],
                int_rx: aio.ChanReceiver[bool],
            ):
                """
                Playout the synthesized audio, & handle interruptions.
                """

                def _interp(a: float, b: float, t: float) -> float:
                    return a + (b - a) * (3 * t**2 - 2 * t**3)

                nonlocal agent_speaking
                await po_validate.wait()

                sample_rate = 24000  # TODO(theomonnom): rm hardcoded

                # synthesis validated, start playout
                try:
                    agent_speaking = True
                    int_status = False
                    int_samples = 0
                    samples = 0

                    select = aio.select([po_rx, int_rx])
                    async for s in select:
                        if s.selected is po_rx:
                            if isinstance(s.exc, aio.ChanClosed):
                                break
                            frame: rtc.AudioFrame = s.result()
                            data = frame.data

                            for i in range(0, len(data)):
                                samples += 1
                                e = data[i] / 32768

                                if int_status:
                                    t_vol = self._base_volume * self._int_volume
                                    e *= max(
                                        _interp(
                                            self._base_volume,
                                            t_vol,
                                            samples / int_samples,
                                        ),
                                        t_vol,
                                    )

                                data[i] = int(e * 32768)

                            await audio_source.capture_frame(frame)

                        if s.selected is int_rx:
                            int_status = s.result()
                            int_samples = samples + int(
                                self._int_volume_duration * sample_rate
                            )

                    # TODO everything is played out, should we update the chat ctx here?
                finally:
                    agent_speaking = False

            chat_ctx = evolve(self._chat_ctx)
            chat_ctx.messages.append(
                allm.ChatMessage(role=allm.ChatRole.USER, text=text)
            )

            po_tx, po_rx = aio.channel()
            int_tx, int_rx = aio.channel()
            synth_task = asyncio.create_task(self._synthesize(chat_ctx, po_tx))
            play_task = asyncio.create_task(_playout_task(chat_ctx, po_rx, int_rx))

        async def _cancel_interruption():
            """
            Cancel the interruption task, interruption isn't needed anymore.
            """
            logger.debug("_cancel_interruption")
            nonlocal int_task, int_tx
            if int_task is not None:
                int_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await int_task

                int_task = None

        async def _start_interruption():
            """
            Notify the playout task we want to smoothly interrupt the agent voice.
            """
            logger.debug("_start_interruption")
            nonlocal int_task, int_tx
            if int_tx is None or int_task is not None:
                return

            try:
                await asyncio.sleep(self._int_speech_duration)
                int_tx.send_nowait(True)
                await asyncio.sleep(self._int_speech_validation)
            except asyncio.CancelledError:
                int_tx.send_nowait(False)

        select = aio.select([vad_stream, stt_stream, audio_stream])
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
                        if agent_speaking:
                            await _start_interruption()
                        else:
                            await _cancel_synthesis()

                    elif vad_event.type == avad.VADEventType.END_OF_SPEECH:
                        po_validate.notify()
                        await _cancel_interruption()

                elif s.selected is stt_stream:
                    # handle STT events
                    stt_event: astt.SpeechEvent = s.result()
                    if stt_event.type == astt.SpeechEventType.FINAL_TRANSCRIPT:
                        alt = stt_event.alternatives[0]
                        user_speech += f"{alt.text} "

                        if not agent_speaking:
                            await _start_synthesis(user_speech)
                        else:
                            await _start_interruption()

                    elif stt_event.type == astt.SpeechEventType.END_OF_SPEECH:
                        po_validate.notify()

    async def _synthesize(
        self, chat_ctx: allm.ChatContext, po_tx: aio.ChanSender[rtc.AudioFrame]
    ) -> None:
        """
        Do LLM inference and TTS synthesis.
        """
        tts_stream = self._tts.stream()

        async def _llm_gen():
            stream: allm.LLMStream | None = None
            try:
                stream = await self._llm.chat(chat_ctx, self._fnc_ctx)
                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    tts_stream.push_text(delta.content)

                tts_stream.mark_segment_end()
            except asyncio.CancelledError:
                if stream is not None:
                    # wait=False will cancel running functions
                    await asyncio.shield(stream.aclose(wait=False))

        async def _tts_gen():
            try:
                async for event in tts_stream:
                    if event.type == atts.SynthesisEventType.FINISHED:
                        break

                    if event.type != atts.SynthesisEventType.AUDIO:
                        continue

                    assert event.audio
                    po_tx.send_nowait(event.audio.data)
            except asyncio.CancelledError:
                await tts_stream.aclose(wait=False)
            finally:
                po_tx.close()

        await asyncio.gather(_llm_gen(), _tts_gen())
