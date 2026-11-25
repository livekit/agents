# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from livekit import rtc
from livekit.agents import vad as vad_module
from livekit.agents.utils import is_given

from .log import logger

if TYPE_CHECKING:
    from .stt import STTOptions


class TurnDetection:
    """Guard forced segment commits and uncommitted partials using local VAD."""

    def __init__(
        self, *, opts: STTOptions, vad: vad_module.VAD | None, on_end_of_speech: Callable[[], None]
    ) -> None:
        self._opts = opts
        self._vad = vad
        self._emit_end_of_speech = on_end_of_speech
        self._vad_stream: vad_module.VADStream | None = None
        self._vad_close_task: asyncio.Task[None] | None = None
        self._eos_timeout_handle: asyncio.TimerHandle | None = None
        self._partial_timeout_handle: asyncio.TimerHandle | None = None
        self._reset()

    def _reset(self) -> None:
        self._vad_speaking = False
        self._audio_duration_since_commit = 0.0
        self._num_pending_client_commits = 0
        self._last_turn_probability: float | None = None
        self._last_partial_text = ""
        self._eos_pending = False
        self._partial_pending = False
        self._partial_started_turn = False
        self._num_text_updates = 0
        self._pending_commit_results: deque[tuple[bool, int]] = deque()

    async def start(self) -> None:
        """Reset turn state and start local VAD for a WebSocket connection."""
        await self.aclose()
        self._reset()
        if (
            self._vad is not None
            and self._opts.model_id == "scribe_v3_live"
            and self._opts.resolved_commit_strategy != "manual"
        ):
            try:
                self._vad_stream = self._vad.stream()
            except Exception:
                logger.warning(
                    "local VAD failed; disabling forced-commit guard for this connection",
                    exc_info=True,
                )

    async def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._audio_duration_since_commit += frame.duration
        if self._vad_stream is None:
            return
        try:
            self._vad_stream.push_frame(frame)
        except Exception:
            await self._disable_vad()

    async def end_input(self) -> None:
        if self._vad_stream is None:
            return
        try:
            self._vad_stream.end_input()
        except Exception:
            await self._disable_vad()

    async def run_vad(self) -> None:
        stream = self._vad_stream
        if stream is None:
            return
        try:
            async for event in stream:
                if stream is not self._vad_stream:
                    break
                if event.type == vad_module.VADEventType.START_OF_SPEECH:
                    self._vad_speaking = True
                    self._cancel_partial_timeout()
                elif event.type == vad_module.VADEventType.END_OF_SPEECH:
                    self._vad_speaking = False
                    self._start_partial_timeout()
        except Exception:
            await self._disable_vad()
        else:
            # On cancellation, aclose() finishes the pending turn before closing VAD.
            if stream is self._vad_stream:
                await self._close_vad()

    async def _disable_vad(self) -> None:
        if self._vad_stream is None:
            return
        logger.warning(
            "local VAD failed; disabling forced-commit guard for this connection", exc_info=True
        )
        await self._close_vad()

    async def _close_vad(self) -> None:
        self._cancel_partial_timeout()
        self._partial_pending = False
        self._partial_started_turn = False
        stream, self._vad_stream = self._vad_stream, None
        self._vad_speaking = False
        if stream is not None:
            self._vad_close_task = asyncio.create_task(self._close_vad_stream(stream))
        task = self._vad_close_task
        if task is None:
            return
        try:
            # Teardown must be able to join cleanup if its first caller is cancelled.
            await asyncio.shield(task)
        finally:
            if task.done() and self._vad_close_task is task:
                self._vad_close_task = None

    async def _close_vad_stream(self, stream: vad_module.VADStream) -> None:
        try:
            await stream.aclose()
        except Exception:
            logger.warning("failed to close local VAD", exc_info=True)

    def on_commit_requested(self) -> None:
        self._num_pending_client_commits += 1

    def end_speech(self) -> None:
        self._eos_pending = False
        self._partial_pending = False
        self._partial_started_turn = False
        self.cancel_eos_timeout()
        self._cancel_partial_timeout()
        self._emit_end_of_speech()

    def on_partial_transcript(
        self, probability: float | None, *, text: str, starts_turn: bool = False
    ) -> bool:
        self._last_turn_probability = probability
        text_changed = bool(text and text != self._last_partial_text)
        if text_changed:
            self._last_partial_text = text
            self._num_text_updates += 1
            # New segment text still needs its own final transcript.
            self.cancel_eos_timeout()
            self._partial_pending = True
            if starts_turn:
                self._partial_started_turn = True
            self._start_partial_timeout()
        return text_changed

    def on_committed_transcript(self, data: dict[str, Any], *, final_message_type: str) -> bool:
        """Return whether this final transcript is a suspected forced commit.

        Classify once per commit and retain the result for its delayed copy.
        """
        message_type = data.get("message_type")
        has_text = bool(data.get("text"))
        result: tuple[bool, int] | None = None
        if message_type == "committed_transcript":
            result = self._classify_commit(has_text=has_text)
            if final_message_type == "committed_transcript_with_timestamps":
                self._pending_commit_results.append(result)
        elif message_type == "committed_transcript_with_timestamps":
            if final_message_type == message_type:
                result = (
                    self._pending_commit_results.popleft()
                    if self._pending_commit_results
                    else self._classify_commit(has_text=has_text)
                )

        if result is not None and result[1] == self._num_text_updates:
            self._partial_pending = False
            self._partial_started_turn = False
            self._cancel_partial_timeout()

        if result is None:
            return False
        is_forced_commit, num_text_updates = result
        if message_type != final_message_type or not is_forced_commit or not has_text:
            return False
        self._eos_pending = True
        if num_text_updates == self._num_text_updates or not self._partial_pending:
            self._start_eos_timeout()
        return True

    def _classify_commit(self, *, has_text: bool) -> tuple[bool, int]:
        # The provider's ~36s segment limit also applies to automatic strategies.
        # This is a heuristic: a silence commit can coincide with that boundary.
        is_forced_commit = (
            self._opts.model_id == "scribe_v3_live"
            and self._opts.resolved_commit_strategy != "manual"
            and not self._num_pending_client_commits
            and 34.0 <= self._audio_duration_since_commit <= 38.0
            and self._last_turn_probability is not None
            and self._last_turn_probability < 0.9
            and self._vad_speaking
        )
        if is_forced_commit and has_text:
            logger.debug(
                "deferring end of speech for a suspected forced commit",
                extra={
                    "audio_duration_since_commit": self._audio_duration_since_commit,
                    "turn_probability": self._last_turn_probability,
                    "vad_speaking": self._vad_speaking,
                },
            )
        self._audio_duration_since_commit = 0.0
        self._num_pending_client_commits = max(0, self._num_pending_client_commits - 1)
        self._last_turn_probability = None
        self._last_partial_text = ""
        return is_forced_commit, self._num_text_updates

    def cancel_eos_timeout(self) -> None:
        if self._eos_timeout_handle is not None:
            self._eos_timeout_handle.cancel()
            self._eos_timeout_handle = None

    def _start_eos_timeout(self) -> None:
        self.cancel_eos_timeout()
        timeout = self._eos_timeout_duration
        self._eos_timeout_handle = asyncio.get_running_loop().call_later(
            timeout, self._on_eos_timeout, timeout
        )

    @property
    def _eos_timeout_duration(self) -> float:
        server_vad = self._opts.server_vad if is_given(self._opts.server_vad) else None
        silence_threshold = (server_vad or {}).get("vad_silence_threshold_secs")
        # The provider may send no further transcript after a forced commit.
        # Allow a short grace period beyond its silence threshold before EOS.
        return (silence_threshold if silence_threshold is not None else 1.5) + 0.5

    def _on_eos_timeout(self, timeout: float) -> None:
        self._eos_timeout_handle = None
        logger.warning(
            "emitting end of speech after a suspected forced commit because no continuation transcript arrived",
            extra={
                "eos_timeout": timeout,
                "audio_duration_since_commit": self._audio_duration_since_commit,
                "vad_speaking": self._vad_speaking,
            },
        )
        self.end_speech()

    def _start_partial_timeout(self) -> None:
        if (
            self._partial_timeout_handle is not None
            or not self._partial_started_turn
            or not self._partial_pending
            or self._vad_stream is None
            or self._vad_speaking
        ):
            return
        timeout = self._eos_timeout_duration
        self._partial_timeout_handle = asyncio.get_running_loop().call_later(
            timeout, self._on_partial_timeout, timeout
        )

    def _on_partial_timeout(self, timeout: float) -> None:
        self._partial_timeout_handle = None
        logger.warning(
            "emitting end of speech for an uncommitted partial while local VAD remains silent",
            extra={"eos_timeout": timeout},
        )
        self.end_speech()

    def _cancel_partial_timeout(self) -> None:
        if self._partial_timeout_handle is not None:
            self._partial_timeout_handle.cancel()
            self._partial_timeout_handle = None

    async def aclose(self) -> None:
        # End any deferred turn before discarding its connection state.
        if self._eos_pending or (
            self._partial_started_turn and self._partial_pending and self._vad_stream is not None
        ):
            self.end_speech()
        self.cancel_eos_timeout()
        self._cancel_partial_timeout()
        await self._close_vad()
