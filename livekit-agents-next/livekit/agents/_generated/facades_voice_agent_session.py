# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, cast

from .. import _bindings, _core
from ..types import NOT_GIVEN
from ..utils import is_given
from ..voice._session_bindings import _stt_model, _tts_model
from ..voice._session_callbacks import _items, _origin
from ..voice.io import SpeechHandle

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Sequence

    from livekit.agents._generated.expressive import ExpressiveOptions
    from livekit.agents.llm.tool_context import ToolChoice

    from ..llm import ChatContext
    from ..stt import STT
    from ..tts import TTS
    from ..types import NotGivenOr
    from ..voice import Agent
    from .chat_items import ChatItem
    from .expressive import ExpressiveOptions as HostExpressiveOptions
    from .rtc_frames import AudioFrame
    from .session_options import ResolvedSessionOptions, TextTransforms
    from .turn import TurnHandlingOptions
    from .usage import Usage
    from .vad import VAD


class VoiceSession:
    """`voice.VoiceSession`'s commands: the owner's settings and reads, and the running session's
    commands, which refuse before `start()`.
    """

    if TYPE_CHECKING:
        @property
        def _running(self) -> _core.VoiceSession: ...

        @property
        def _options(self) -> _core.SessionOptions: ...

        _native: _core._core_SessionOwner

    async def aclose(self) -> None:
        """Close the session and wait for it."""
        await self._native.aclose()

    def shutdown(self, *, drain: bool = True) -> None:
        """`aclose()` without waiting for it; `drain` lets queued speech finish first."""
        self._native.shutdown(drain=drain)

    @contextlib.asynccontextmanager
    async def _claim_user_turn(self) -> AsyncIterator[None]:
        """A programmatic user turn: the user state is pinned to `speaking` and `wait_for_idle`
        held while inside.
        """
        target = self._running
        target.claim_user_turn()
        try:
            yield
        finally:
            target.release_user_turn()

    @property
    def stt(self) -> STT | None:
        """The STT the session listens with now: an `override_options` layer's while one holds it,
        else the base's (the constructor's or `update_options`').
        """
        return cast("STT | None", self._options.stt)

    @property
    def tts(self) -> TTS | None:
        """The TTS the session speaks with now, resolved as `stt` is."""
        return cast("TTS | None", self._options.tts)

    @property
    def vad(self) -> VAD | None:
        """The VAD the session runs now, resolved as `stt` is."""
        return cast("VAD | None", self._options.vad)

    @property
    def current_agent(self) -> Agent:
        """The agent driving the session; refused before `start()` unless `update_agent` named one."""
        return cast("Agent", self._native.current_agent)

    @property
    def usage(self) -> Usage:
        """Empty before startup; the final snapshot remains readable after close."""
        return self._native.usage

    @property
    def keyterms(self) -> list[str]:
        """No keyterms are applied before a voice driver exists."""
        return self._native.keyterms

    @property
    def options(self) -> ResolvedSessionOptions:
        """The configuration the core runs on, current the instant a change is sent."""
        return self._native.resolved_options(cast(Any, self)._adapter._legs())

    def update_options(
        self,
        *,
        stt: NotGivenOr[STT | str | None] = NOT_GIVEN,
        tts: NotGivenOr[TTS | str | None] = NOT_GIVEN,
        vad: NotGivenOr[VAD | None] = NOT_GIVEN,
        turn_handling: NotGivenOr[TurnHandlingOptions] = NOT_GIVEN,
        min_consecutive_speech_delay: NotGivenOr[float] = NOT_GIVEN,
        tts_text_transforms: NotGivenOr[Sequence[TextTransforms] | None] = NOT_GIVEN,
        expressive: NotGivenOr[bool | HostExpressiveOptions] = NOT_GIVEN,
        keyterms: NotGivenOr[Sequence[str]] = NOT_GIVEN,
    ) -> None:
        """`update_options(..)`: change only the supplied settings. A replaced `stt`, `tts` or `vad`
        is installed in place on a running session, without restarting the call or interrupting
        speech; a setting an `override_options` layer holds takes the new value once the layer is
        released. `turn_handling` merges by key over the session's, nested groups included; its
        `interruption` takes no `mode`, which is set with `VoiceSession(turn_handling=)`.
        """
        self._native.update_options(
            stt=_stt_model(stt) if is_given(stt) else stt,
            tts=_tts_model(tts) if is_given(tts) else tts,
            vad=vad,
            turn_handling=turn_handling,
            min_consecutive_speech_delay=min_consecutive_speech_delay,
            tts_text_transforms=tts_text_transforms,
            expressive=expressive,
            keyterms=keyterms,
        )

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

    async def set_participant(self, identity: str | None) -> None:
        """`RoomIO.set_participant(identity)`: point the audio input and the user transcript at
        someone else. `None` unlinks.
        """
        await self._running.set_participant(identity)

    def _clear_user_turn(self) -> None:
        """`AgentSession.clear_user_turn()`: drop the open turn and everything accumulated for it."""
        self._running.clear_user_turn()

    def interrupt(self, *, force: bool = False) -> asyncio.Future[None]:
        """`AgentSession.interrupt(force=)`: the current and queued speeches up to the first
        uninterruptible; answers when they are done, or with the refusal.
        """
        return asyncio.ensure_future(self._running.interrupt(force=force))

    async def drain(self) -> None:
        """`AgentSession.drain()`: the driving agent's `on_exit`, then scheduling paused until no
        speech is left. Answers `None`, or why it was refused.
        """
        await self._running.drain()

    async def wait_for_idle(self) -> None:
        """`AgentSession.wait_for_idle()`: answers when the session is idle (no hold but `hold`)."""
        await self._running.wait_for_idle()

    def _commit_user_turn(
        self,
        *,
        transcript_timeout: float = 2.0,
        stt_flush_duration: float = 2.0,
        skip_reply: bool = False,
    ) -> asyncio.Future[str]:
        """`AgentSession.commit_user_turn(..)`: flush, wait for the final, commit, answer with the
        transcript. Runs whether or not awaited; a newer commit supersedes it.
        """
        return asyncio.ensure_future(self._running.commit_user_turn(
            transcript_timeout=transcript_timeout,
            stt_flush_duration=stt_flush_duration,
            skip_reply=skip_reply,
        ))
