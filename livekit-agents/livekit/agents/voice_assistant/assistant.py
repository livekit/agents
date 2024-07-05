from __future__ import annotations

import asyncio
from typing import Any, AsyncIterable, Awaitable, Callable, Literal, Optional, Union

from livekit import rtc

from .. import llm, stt, tokenize, utils, vad
from .. import tts as text_to_speech
from . import impl


async def _default_will_synthesize_assistant_reply(
    assistant: VoiceAssistant, chat_ctx: llm.ChatContext
) -> llm.LLMStream:
    return assistant.llm.chat(chat_ctx=chat_ctx, fnc_ctx=assistant.fnc_ctx)


EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
    "user_speech_committed",
    "agent_speech_committed",
    "agent_speech_interrupted",
    "function_calls_collected",
    "function_calls_finished",
]

WillSynthesizeAssistantReply = Callable[
    ["VoiceAssistant", llm.ChatContext],
    Union[Optional[llm.LLMStream], Awaitable[Optional[llm.LLMStream]]],
]


class VoiceAssistant(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        vad: vad.VAD,
        stt: stt.STT,
        llm: llm.LLM,
        tts: text_to_speech.TTS,
        chat_ctx: llm.ChatContext = llm.ChatContext(),
        fnc_ctx: llm.FunctionContext | None = None,
        allow_interruptions: bool = True,
        interrupt_speech_duration: float = 0.65,
        interrupt_min_words: int = 3,
        preemptive_synthesis: bool = True,
        transcription: bool = True,
        will_synthesize_assistant_reply: WillSynthesizeAssistantReply = _default_will_synthesize_assistant_reply,
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(),
        hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word,
        transcription_speed: float = 3.83,
        debug: bool = False,
        plotting: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        loop = loop or asyncio.get_event_loop()

        def will_synthesize_llm_stream_impl(
            _: impl.AssistantImpl, chat_ctx: llm.ChatContext
        ):
            return will_synthesize_assistant_reply(self, chat_ctx)

        opts = impl.ImplOptions(
            plotting=plotting,
            debug=debug,
            allow_interruptions=allow_interruptions,
            int_speech_duration=interrupt_speech_duration,
            int_min_words=interrupt_min_words,
            preemptive_synthesis=preemptive_synthesis,
            transcription=transcription,
            sentence_tokenizer=sentence_tokenizer,
            word_tokenizer=word_tokenizer,
            hyphenate_word=hyphenate_word,
            transcription_speed=transcription_speed,
            will_synthesize_assistant_answer=will_synthesize_llm_stream_impl,
        )

        # wrap with StreamAdapter automatically when streaming is not supported on a specific TTS
        # to override StreamAdapter options, create the adapter manually
        if not tts.streaming_supported:
            tts = text_to_speech.StreamAdapter(
                tts=tts, sentence_tokenizer=tokenize.basic.SentenceTokenizer()
            )

        self._impl = impl.AssistantImpl(
            vad=vad,
            stt=stt,
            llm=llm,
            tts=tts,
            emitter=self,
            options=opts,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            loop=loop,
        )

    def start(
        self, room: rtc.Room, participant: rtc.RemoteParticipant | str | None = None
    ) -> None:
        """Start the voice assistant

        Args:
            room: the room to use
            participant: the participant to listen to, can either be a participant or a participant identity
                If None, the first participant found in the room will be selected
        """
        self._impl.start(room=room, participant=participant)

    async def say(
        self,
        source: str | llm.LLMStream | AsyncIterable[str],
        *,
        allow_interruptions: bool = True,
        add_to_chat_ctx: bool = True,
    ) -> None:
        """
        Make the assistant say something.
        The source can be a string, an LLMStream or an AsyncIterable[str]

        Args:
            source: the source of the speech
            allow_interruptions: whether the speech can be interrupted
            add_to_chat_ctx: whether to add the speech to the chat context
        """
        await self._impl.say(
            source=source,
            allow_interruptions=allow_interruptions,
            add_to_chat_ctx=add_to_chat_ctx,
        )

    async def aclose(self) -> None:
        """
        Close the voice assistant
        """
        await self._impl.aclose()

    def on(self, event: EventTypes, callback: Callable[[Any], None] | None = None):
        """Register a callback for an event

        Args:
            event: the event to listen to (see EventTypes)
                - user_started_speaking: the user started speaking
                - user_stopped_speaking: the user stopped speaking
                - agent_started_speaking: the agent started speaking
                - agent_stopped_speaking: the agent stopped speaking
                - user_speech_committed: the user speech was committed to the chat context
                - agent_speech_committed: the agent speech was committed to the chat context
                - agent_speech_interrupted: the agent speech was interrupted
                - function_calls_collected: received the complete set of functions to be executed
                - function_calls_finished: all function calls have been completed
            callback: the callback to call when the event is emitted
        """
        return super().on(event, callback)

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None:
        return self._impl._fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, fnc_ctx: llm.FunctionContext | None) -> None:
        self._impl._fnc_ctx = fnc_ctx

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._impl._chat_ctx

    @property
    def llm(self) -> llm.LLM:
        return self._impl._llm

    @property
    def tts(self) -> text_to_speech.TTS:
        return self._impl._tts

    @property
    def stt(self) -> stt.STT:
        return self._impl._stt

    @property
    def vad(self) -> vad.VAD:
        return self._impl._vad
