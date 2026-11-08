# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from .. import _bindings, _core

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ..voice.agent_session import VoiceSession as HostVoiceSession
    from .voice_session import FillerSource


Context_T = TypeVar("Context_T")


class RunContext(Generic[Context_T]):
    """`RunContext`'s hold of the call's floor."""

    if TYPE_CHECKING:
        def _call(self) -> Any: ...

    @contextlib.asynccontextmanager
    async def foreground(self) -> AsyncIterator[HostVoiceSession[Context_T]]:
        """Wait for idle, then hold the floor while interactive work runs (a `say`), so no other
        speech or deferred tool reply lands in between.
        """
        target = self._call()._core
        value = await target.wait_for_idle_and_hold(origin=cast(Any, self)._origin)
        try:
            yield self._call()
        finally:
            target.release_idle_hold(value)

    @contextlib.asynccontextmanager
    async def with_filler(
        self,
        source: FillerSource,
        *,
        delay: float = 0.0,
        interval: float | None = None,
        max_steps: int | None = None,
    ) -> AsyncIterator[None]:
        """Schedule filler speech while a long-running step blocks the tool.

        After `delay` seconds of continuous idle, `source` is spoken (a string) or asked (a
        callable taking the step count, returning text, a handle it spoke itself, or `None` to
        skip this fire); with `interval` set it repeats, up to `max_steps`.
        """
        target = self._call()._core
        value = target.filler_start(
            cast(Any, self).function_call.call_id,
            cast(Any, self).speech_handle.id,
            source,
            delay=delay,
            interval=interval,
            max_steps=max_steps,
            origin=cast(Any, self)._origin,
        )
        try:
            yield
        finally:
            target.filler_stop(value)
