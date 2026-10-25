# SPDX-License-Identifier: Apache-2.0
"""Prism lifecycle interfaces kept local to the Rime plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar, cast

from livekit.agents import llm
from livekit.agents.types import NOT_GIVEN, NotGivenOr


@dataclass
class InputSpeechHintEvent:
    audio_start_ms: int
    gain: float = 0.25
    timeout: float = 0.6


@dataclass
class InputSpeechStartedEvent(llm.InputSpeechStartedEvent):
    interrupt_response: bool = True


@dataclass
class GenerationCreatedEvent(llm.GenerationCreatedEvent):
    playback_session: llm.RealtimeSession[Any] | None = field(
        default=None, repr=False, compare=False
    )


TEvent = TypeVar("TEvent")


class RealtimeSession(llm.RealtimeSession[TEvent], Generic[TEvent]):
    def can_reuse(self, chat_ctx: llm.ChatContext) -> bool:
        """Whether a handoff can keep this conversation with the supplied context.

        Providers may accept a local projection while retaining provider-owned
        history. This does not grant permission to edit remote history.
        """
        return self.capabilities.mutable_chat_context or self.chat_ctx.copy(
            exclude_instructions=True, exclude_handoff=True, exclude_config_update=True
        ).is_equivalent(
            chat_ctx.copy(
                exclude_instructions=True, exclude_handoff=True, exclude_config_update=True
            )
        )

    async def set_input_text(
        self, text: str, *, message_id: NotGivenOr[str] = NOT_GIVEN
    ) -> llm.ChatMessage:
        """Stage a user turn for the immediately following generate_reply call.

        Unlike update_chat_ctx, this explicitly requests a new user turn. The
        default uses the existing history update path.
        """
        chat_ctx = self.chat_ctx.copy()
        message = chat_ctx.add_message(role="user", content=text, id=message_id)
        await self.update_chat_ctx(chat_ctx)
        return message

    def playback_finished(
        self,
        *,
        response_id: str | None,
        message_id: str | None,
        audio_end_ms: int,
        interrupted: bool,
        interrupted_by_user: bool,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
        modalities: list[Literal["text", "audio"]] | None = None,
    ) -> None:
        """Report completed or permanently stopped playback, including skipped audio.

        message_id=None means the response completed without messages or stopped
        before a message was played.
        interrupted_by_user means caller audio, not typed input or an app stop.
        The default retains the previous message truncation behavior.
        """
        if (
            interrupted
            and message_id is not None
            and modalities is not None
            and self.capabilities.message_truncation
        ):
            self.truncate(
                message_id=message_id,
                modalities=modalities or [],
                audio_end_ms=audio_end_ms,
                audio_transcript=audio_transcript,
            )


async def set_input_text(
    session: llm.RealtimeSession[Any], text: str, *, message_id: NotGivenOr[str] = NOT_GIVEN
) -> llm.ChatMessage:
    method = getattr(session, "set_input_text", None)
    if method is not None:
        return cast(llm.ChatMessage, await method(text, message_id=message_id))
    return await RealtimeSession.set_input_text(
        cast(RealtimeSession[Any], session), text, message_id=message_id
    )


def playback_finished(session: llm.RealtimeSession[Any], **kwargs: Any) -> None:
    method = getattr(session, "playback_finished", None)
    if method is not None:
        method(**kwargs)
    else:
        RealtimeSession.playback_finished(cast(RealtimeSession[Any], session), **kwargs)


def can_reuse(session: llm.RealtimeSession[Any], chat_ctx: llm.ChatContext) -> bool:
    method = getattr(session, "can_reuse", None)
    return (
        bool(method(chat_ctx))
        if method is not None
        else RealtimeSession.can_reuse(cast(RealtimeSession[Any], session), chat_ctx)
    )
