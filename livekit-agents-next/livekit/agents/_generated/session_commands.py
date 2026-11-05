# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] declarations in lk-agents-abi.

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .. import _core
from livekit.agents.llm.tool_context import ToolChoice
from collections.abc import Sequence
from ..voice.io import SpeechHandle
from ..types import NOT_GIVEN
from ..types import NotGivenOr
from ..utils import is_given
from ..llm import ChatContext
from ..voice._session_callbacks import _items
from ..voice._session_callbacks import _origin
from collections.abc import AsyncIterable
from .rtc_frames import AudioFrame
from .usage import Usage
from livekit.agents.voice.io import TurnDetectionMode
from .turn import EndpointingOptions
from .turn import InterruptionOptions
from ..stt import STT
from ..voice._session_bindings import _stt_model
from ..tts import TTS
from ..voice._session_bindings import _tts_model
from .vad import VAD
from ..voice._session_bindings import TextTransforms
from .expressive import ExpressiveOptions


class SessionCommands:
    if TYPE_CHECKING:

        @property
        def _owner(self) -> _core._core_SessionOwner: ...

        @property
        def _running(self) -> _core.VoiceSession: ...

    def generate_reply(
        self,
        *,
        user_input: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[Sequence[str]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
    ) -> SpeechHandle:
        """`generate_reply()`: reply now, with no end of turn behind it. The overrides apply to this
        reply only; `tools` narrows the driving agent's tools to these names (its provider-owned
        tools stay), refused on a realtime model as `chat_ctx` is. Sent from the agent's
        `on_enter`, it is that hook's reply.
        """
        return SpeechHandle(
            self._running.generate_reply(
                user_input=user_input if is_given(user_input) else None,
                instructions=instructions if is_given(instructions) else None,
                tool_choice=tool_choice if is_given(tool_choice) else None,
                tools=tools if is_given(tools) else None,
                allow_interruptions=allow_interruptions if is_given(allow_interruptions) else None,
                chat_ctx=_items(chat_ctx) if is_given(chat_ctx) else None,
                origin=_origin(),
            )
        )

    async def set_participant(self, identity: str | None) -> None:
        """`RoomIO.set_participant(identity)`: point the audio input and the user transcript at
        someone else. `None` unlinks.
        """
        return await self._running.set_participant(identity)

    def clear_user_turn(self) -> None:
        """`AgentSession.clear_user_turn()`: drop the open turn and everything accumulated for it."""
        return self._running.clear_user_turn()

    def interrupt(self, *, force: bool = False) -> asyncio.Future[None]:
        """`AgentSession.interrupt(force=)`: the current and queued speeches up to the first
        uninterruptible; answers when they are done, or with the refusal.
        """
        return asyncio.ensure_future(self._running.interrupt(force=force))

    async def drain(self) -> None:
        """`AgentSession.drain()`: the driving agent's `on_exit`, then scheduling paused until no
        speech is left. Answers `None`, `"activity_changing"` or `"closing"`.
        """
        await self._running.drain()

    async def wait_for_idle(self) -> None:
        """`AgentSession.wait_for_idle()`: answers when the session is idle (no hold but `hold`)."""
        return await self._running.wait_for_idle()

    def say(
        self,
        text: str | AsyncIterable[str],
        *,
        audio: NotGivenOr[AsyncIterable[AudioFrame]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        add_to_chat_ctx: bool = True,
    ) -> SpeechHandle:
        """`say()`: speak `text` without the LLM, the words given or a stream of them read to its
        end. With `audio`, those frames play instead of the TTS and `text`, a `str`, is still
        committed. The handle is the promise the session fulfils.
        """
        return SpeechHandle(
            self._running.say(
                text,
                audio=audio if is_given(audio) else None,
                allow_interruptions=allow_interruptions if is_given(allow_interruptions) else None,
                add_to_chat_ctx=add_to_chat_ctx,
                origin=_origin(),
            )
        )

    def commit_user_turn(
        self,
        *,
        transcript_timeout: float = 2.0,
        stt_flush_duration: float = 2.0,
        skip_reply: bool = False,
    ) -> asyncio.Future[str]:
        """`AgentSession.commit_user_turn(..)`: flush, wait for the final, commit, answer with the
        transcript. Runs whether or not awaited; a newer commit supersedes it.
        """
        return asyncio.ensure_future(
            self._running.commit_user_turn(
                transcript_timeout=transcript_timeout,
                stt_flush_duration=stt_flush_duration,
                skip_reply=skip_reply,
            )
        )

    @property
    def usage(self) -> Usage:
        """Empty before startup; the final snapshot remains readable after close."""
        return self._owner.usage

    @property
    def keyterms(self) -> list[str]:
        """No keyterms are applied before a voice driver exists."""
        return self._owner.keyterms

    def update_options(
        self,
        *,
        stt: NotGivenOr[STT | str | None] = NOT_GIVEN,
        tts: NotGivenOr[TTS | str | None] = NOT_GIVEN,
        vad: NotGivenOr[VAD | None] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        endpointing_opts: NotGivenOr[EndpointingOptions] = NOT_GIVEN,
        interruption: NotGivenOr[InterruptionOptions] = NOT_GIVEN,
        min_consecutive_speech_delay: NotGivenOr[float] = NOT_GIVEN,
        tts_text_transforms: NotGivenOr[Sequence[TextTransforms] | None] = NOT_GIVEN,
        expressive: NotGivenOr[bool | ExpressiveOptions] = NOT_GIVEN,
        keyterms: NotGivenOr[Sequence[str]] = NOT_GIVEN,
    ) -> None:
        """`update_options(..)`: change only the supplied settings. A replaced `stt`, `tts` or `vad`
        is installed in place on a running session, without restarting the call or interrupting
        speech; a setting an `override_options` layer holds takes the new value once the layer is
        released. `interruption` merges over the session's and takes no `mode`, which is set with
        `VoiceSession(turn_handling=)`.
        """
        return self._owner.update_options(
            stt=_stt_model(stt) if is_given(stt) else stt,
            tts=_tts_model(tts) if is_given(tts) else tts,
            vad=vad,
            turn_detection=turn_detection,
            endpointing_opts=endpointing_opts,
            interruption=interruption,
            min_consecutive_speech_delay=min_consecutive_speech_delay,
            tts_text_transforms=tts_text_transforms,
            expressive=expressive,
            keyterms=keyterms,
        )
