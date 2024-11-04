from __future__ import annotations

import asyncio
from typing import AsyncIterable

from .. import utils
from ..llm import LLMStream
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
    ) -> None:
        self._id = id
        self._allow_interruptions = allow_interruptions
        self._add_to_chat_ctx = add_to_chat_ctx

        # is_reply is True when the speech is answering to a user question
        self._is_reply = is_reply
        self._user_question = user_question
        self._user_commited = False

        self._init_fut: asyncio.Future[None] = asyncio.Future()
        self._initialized = False
        self._speech_commited = False  # speech committed (interrupted or not)

        # source and synthesis_handle are None until the speech is initialized
        self._source: str | LLMStream | AsyncIterable[str] | None = None
        self._synthesis_handle: SynthesisHandle | None = None

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

    def mark_user_commited(self) -> None:
        self._user_commited = True

    def mark_speech_commited(self) -> None:
        self._speech_commited = True

    @property
    def user_commited(self) -> bool:
        return self._user_commited

    @property
    def speech_commited(self) -> bool:
        return self._speech_commited

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

    def interrupt(self) -> None:
        if not self.allow_interruptions:
            raise RuntimeError("interruptions are not allowed")
        self.cancel()

    def cancel(self) -> None:
        self._init_fut.cancel()

        if self._synthesis_handle is not None:
            self._synthesis_handle.interrupt()
