import asyncio
import logging
import uuid
from typing import Callable, List

from livekit import agents, rtc
from livekit.agents.llm import ChatContext, ChatMessage, ChatRole
from livekit.plugins.elevenlabs import TTS
from livekit.plugins.openai import LLM


class InferenceJob:
    def __init__(
        self,
        transcription: str,
        audio_source: rtc.AudioSource,
        chat_history: List[ChatMessage],
        on_agent_response: Callable[[str, bool], None],
        on_agent_speaking: Callable[[bool], None],
        force_text_response: str | None = None,
    ):
        self._id = uuid.uuid4()
        self._audio_source = audio_source
        self._transcription = transcription
        self._current_response = ""
        self._chat_history = chat_history
        self._tts = TTS(model_id="eleven_turbo_v2")
        self._tts_stream = self._tts.stream()
        self._llm = LLM()
        self._run_task = asyncio.create_task(self._run())
        self._output_queue = asyncio.Queue[rtc.AudioFrame | None]()
        self._speaking = False
        self._finished_generating = False
        self._on_agent_response = on_agent_response
        self._on_agent_speaking = on_agent_speaking
        self._done_future = asyncio.Future()
        self._cancelled = False
        self._force_text_response = force_text_response

    @property
    def id(self):
        return self._id

    @property
    def transcription(self):
        return self._transcription

    @property
    def current_response(self):
        return self._current_response

    @current_response.setter
    def current_response(self, value: str):
        self._current_response = value
        if not self._cancelled:
            self._on_agent_response(value, self._finished_generating)

    @property
    def finished_generating(self):
        return self._finished_generating

    @finished_generating.setter
    def finished_generating(self, value: bool):
        self._finished_generating = value
        if not self._cancelled:
            self._on_agent_response(self.current_response, value)

    async def acancel(self):
        logging.info("Cancelling inference job")
        self._cancelled = True
        self._run_task.cancel()
        await self._done_future
        logging.info("Cancelled inference job")

    @property
    def speaking(self):
        return self._speaking

    @speaking.setter
    def speaking(self, value: bool):
        if value == self._speaking:
            return
        self._speaking = value
        if not self._cancelled:
            self._on_agent_speaking(value)

    async def _run(self):
        logging.info(
            "Running inference with user transcription: %s", self.transcription
        )
        try:
            await asyncio.gather(
                self._llm_task(),
                self._tts_task(),
                asyncio.shield(self._audio_capture_task()),
            )
        except asyncio.CancelledError:
            # Flush audio packets
            while True:
                try:
                    self._output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._output_queue.put_nowait(None)
        except Exception as e:
            logging.exception("Exception in inference %s", e)

    async def _llm_task(self):
        if self._force_text_response:
            self._tts_stream.push_text(self._force_text_response)
            await self._tts_stream.flush()
            return

        chat_context = ChatContext(
            messages=self._chat_history
            + [ChatMessage(role=ChatRole.USER, text=self.transcription)]
        )
        async for chunk in await self._llm.chat(history=chat_context):
            delta = chunk.choices[0].delta.content
            if delta is None:
                break
            self._tts_stream.push_text(delta)
            self.current_response += delta
        self.finished_generating = True
        await self._tts_stream.flush()

    async def _tts_task(self):
        async for event in self._tts_stream:
            if event.type == agents.tts.SynthesisEventType.AUDIO:
                await self._output_queue.put(
                    event.audio.data if event.audio else event.audio
                )
            elif event.type == agents.tts.SynthesisEventType.FINISHED:
                break
        await self._output_queue.put(None)

    async def _audio_capture_task(self):
        while True:
            audio_frame = await self._output_queue.get()
            if audio_frame is None:
                break
            self.speaking = True
            await self._audio_source.capture_frame(audio_frame)
        self.speaking = False
        self._done_future.set_result(True)
        await self._tts_stream.aclose()
