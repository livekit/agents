# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core


class PlayHandle:
    """`voice.PlayHandle`: one sound `BackgroundAudioPlayer.play()` started; awaiting it waits for
    its playout.
    """

    def __init__(self, native: _core.PlayState) -> None:
        self._native = native

    def done(self) -> bool:
        """Whether the sound finished playing."""
        return self._native.done

    def stop(self) -> None:
        """Stop the sound. A source started with a `fade_out` fades out first and stays not done
        until its tail played; with none, playback is cut at once.
        """
        self._native.stop()

    async def wait_for_playout(self) -> None:
        """Wait for the sound to finish playing."""
        await self._native.wait_for_playout()
