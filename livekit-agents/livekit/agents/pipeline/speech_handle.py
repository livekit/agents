from __future__ import annotations

import asyncio
from typing import AsyncIterable

from .. import utils
from ..llm import ChatMessage, LLMStream
from .agent_output import SynthesisHandle


class SpeechHandle:
    def __init__(
        self,
        *,
        id: str,
        allow_interruptions: bool,
        add_to_chat_ctx: bool,
        is_reply: bool,
        user_question: str,
        fnc_nested_depth: int = 0,
        extra_tools_messages: list[ChatMessage] | None = None,
        fnc_text_message_id: str | None = None,
    ) -> None:
        self._id = id
        self._allow_interruptions = allow_interruptions
        self._add_to_chat_ctx = add_to_chat_ctx

        # is_reply is True when the speech is answering to a user question
        self._is_reply = is_reply
        self._user_question = user_question
        self._user_committed = False

        self._init_fut = asyncio.Future[None]()
        self._done_fut = asyncio.Future[None]()
        self._initialized = False
        self._speech_committed = False  # speech committed (interrupted or not)

        # source and synthesis_handle are None until the speech is initialized
        self._source: str | LLMStream | AsyncIterable[str] | None = None
        self._synthesis_handle: SynthesisHandle | None = None

        # nested speech handle and function calls
        self._fnc_nested_depth = fnc_nested_depth
        self._fnc_extra_tools_messages: list[ChatMessage] | None = extra_tools_messages
        self._fnc_text_message_id: str | None = fnc_text_message_id

        self._nested_speech_handles: list[SpeechHandle] = []
        self._nested_speech_changed = asyncio.Event()
        self._nested_speech_done_fut = asyncio.Future[None]()

    @staticmethod
    def create_assistant_reply(
        *,
        allow_interruptions: bool,
        add_to_chat_ctx: bool,
        user_question: str,
    ) -> SpeechHandle:
        return SpeechHandle(
            id=utils.shortuuid(),
            allow_interruptions=allow_interruptions,
            add_to_chat_ctx=add_to_chat_ctx,
            is_reply=True,
            user_question=user_question,
        )

    @staticmethod
    def create_assistant_speech(
        *,
        allow_interruptions: bool,
        add_to_chat_ctx: bool,
    ) -> SpeechHandle:
        return SpeechHandle(
            id=utils.shortuuid(),
            allow_interruptions=allow_interruptions,
            add_to_chat_ctx=add_to_chat_ctx,
            is_reply=False,
            user_question="",
        )

    @staticmethod
    def create_tool_speech(
        *,
        allow_interruptions: bool,
        add_to_chat_ctx: bool,
        fnc_nested_depth: int,
        extra_tools_messages: list[ChatMessage],
        fnc_text_message_id: str | None = None,
    ) -> SpeechHandle:
        return SpeechHandle(
            id=utils.shortuuid(),
            allow_interruptions=allow_interruptions,
            add_to_chat_ctx=add_to_chat_ctx,
            is_reply=False,
            user_question="",
            fnc_nested_depth=fnc_nested_depth,
            extra_tools_messages=extra_tools_messages,
            fnc_text_message_id=fnc_text_message_id,
        )

    async def wait_for_initialization(self) -> None:
        await asyncio.shield(self._init_fut)

    def initialize(
        self,
        *,
        source: str | LLMStream | AsyncIterable[str],
        synthesis_handle: SynthesisHandle,
    ) -> None:
        if self.interrupted:
            raise RuntimeError("speech is interrupted")

        self._source = source
        self._synthesis_handle = synthesis_handle
        self._initialized = True
        self._init_fut.set_result(None)

    def mark_user_committed(self) -> None:
        self._user_committed = True

    def mark_speech_committed(self) -> None:
        self._speech_committed = True

    @property
    def user_committed(self) -> bool:
        return self._user_committed

    @property
    def speech_committed(self) -> bool:
        return self._speech_committed

    @property
    def id(self) -> str:
        return self._id

    @property
    def allow_interruptions(self) -> bool:
        return self._allow_interruptions

    @property
    def add_to_chat_ctx(self) -> bool:
        return self._add_to_chat_ctx

    @property
    def source(self) -> str | LLMStream | AsyncIterable[str]:
        if self._source is None:
            raise RuntimeError("speech not initialized")
        return self._source

    @property
    def synthesis_handle(self) -> SynthesisHandle:
        if self._synthesis_handle is None:
            raise RuntimeError("speech not initialized")
        return self._synthesis_handle

    @synthesis_handle.setter
    def synthesis_handle(self, synthesis_handle: SynthesisHandle) -> None:
        """synthesis handle can be replaced for the same speech.
        This is useful when we need to do a new generation. (e.g for automatic function call answers)"""
        if self._synthesis_handle is None:
            raise RuntimeError("speech not initialized")

        self._synthesis_handle = synthesis_handle

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def is_reply(self) -> bool:
        return self._is_reply

    @property
    def user_question(self) -> str:
        return self._user_question

    @property
    def interrupted(self) -> bool:
        return self._init_fut.cancelled() or (
            self._synthesis_handle is not None and self._synthesis_handle.interrupted
        )

    def join(self) -> asyncio.Future:
        return self._done_fut

    def _set_done(self) -> None:
        self._done_fut.set_result(None)

    def interrupt(self) -> None:
        if not self.allow_interruptions:
            raise RuntimeError("interruptions are not allowed")
        self.cancel()

    def cancel(self, cancel_nested: bool = False) -> None:
        self._init_fut.cancel()

        if self._synthesis_handle is not None:
            self._synthesis_handle.interrupt()

        if cancel_nested:
            for speech in self._nested_speech_handles:
                speech.cancel(cancel_nested=True)
            self.mark_nested_speech_done()

    @property
    def fnc_nested_depth(self) -> int:
        return self._fnc_nested_depth

    @property
    def extra_tools_messages(self) -> list[ChatMessage] | None:
        return self._fnc_extra_tools_messages

    @property
    def fnc_text_message_id(self) -> str | None:
        return self._fnc_text_message_id

    def add_nested_speech(self, speech_handle: SpeechHandle) -> None:
        self._nested_speech_handles.append(speech_handle)
        self._nested_speech_changed.set()

    @property
    def nested_speech_handles(self) -> list[SpeechHandle]:
        return self._nested_speech_handles

    @property
    def nested_speech_changed(self) -> asyncio.Event:
        return self._nested_speech_changed

    @property
    def nested_speech_done(self) -> bool:
        return self._nested_speech_done_fut.done()

    def mark_nested_speech_done(self) -> None:
        if self._nested_speech_done_fut.done():
            return
        self._nested_speech_done_fut.set_result(None)
