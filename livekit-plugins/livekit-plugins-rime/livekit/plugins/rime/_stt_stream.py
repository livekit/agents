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
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, APIStatusError, stt, utils, vad
from livekit.agents.language import LanguageCode
from livekit.agents.types import NotGivenOr

if TYPE_CHECKING:
    from .stt import STT


@dataclass(eq=False)
class _Utterance:
    audio: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    audio_duration: float = 0.0
    speech_start_time: float = field(default_factory=time.time)
    speech_end_time: float | None = None
    publishing: bool = False
    interim: stt.SpeechEvent | None = None
    final: stt.SpeechEvent | None = None
    usage: stt.SpeechEvent | None = None


class SpeechStream(stt.RecognizeStream):
    """Upload utterances concurrently and publish transcripts in speech order."""

    def __init__(
        self, owner: STT, detector: vad.VAD, language: NotGivenOr[str], options: APIConnectOptions
    ) -> None:
        super().__init__(
            stt=owner,
            sample_rate=16000,
            conn_options=APIConnectOptions(timeout=options.timeout, max_retry=0),
        )
        self._owner = owner
        self._vad = detector
        self._language = language

    async def _run(self) -> None:
        failure: asyncio.Future[BaseException] = asyncio.get_running_loop().create_future()
        requests: set[asyncio.Task[None]] = set()
        pending: set[_Utterance] = set()
        ordered: asyncio.Queue[_Utterance | None] = asyncio.Queue()
        pending_bytes = 0

        async def audio_source(utterance: _Utterance) -> AsyncIterator[bytes]:
            nonlocal pending_bytes
            while (chunk := await utterance.audio.get()) is not None:
                pending_bytes -= len(chunk)
                yield chunk

        def publish(utterance: _Utterance, event: stt.SpeechEvent) -> None:
            if utterance.publishing:
                self._event_ch.send_nowait(event)
            elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                # LiveKit snapshots replace earlier text. While an older request is
                # pending, retain only the newest snapshot, never every fragment.
                utterance.interim = event
            elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                utterance.final = event
            else:
                utterance.usage = event

        async def transcribe(utterance: _Utterance, options: dict[str, Any]) -> None:
            text = ""
            try:
                async with self._owner._client.stt.stream(
                    audio_source(utterance), **options
                ) as stream:
                    async for update in stream:
                        kind = str(update.kind)
                        if kind == "delta":
                            text += update.text
                        else:
                            text = update.text
                        final = kind == "final"
                        publish(
                            utterance,
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT
                                if final
                                else stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                request_id=stream.request_id or "",
                                alternatives=[
                                    stt.SpeechData(
                                        text=text,
                                        language=LanguageCode(
                                            getattr(update, "language", options["language"])
                                        ),
                                    )
                                ],
                                speech_end_time=utterance.speech_end_time if final else None,
                            ),
                        )
                    publish(
                        utterance,
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.RECOGNITION_USAGE,
                            request_id=stream.request_id or "",
                            recognition_usage=stt.RecognitionUsage(
                                audio_duration=utterance.audio_duration
                            ),
                        ),
                    )
                utterance.done.set()
            except self._owner._sdk.RimeError as exc:
                raise APIStatusError(
                    "Rime streaming STT failed",
                    request_id=exc.request_id,
                    retryable=False,
                ) from None

        def finished(task: asyncio.Task[None]) -> None:
            requests.discard(task)
            if (
                not task.cancelled()
                and (exc := task.exception()) is not None
                and not failure.done()
            ):
                failure.set_result(exc)

        async def emit_results() -> None:
            while (utterance := await ordered.get()) is not None:
                # Boundaries and transcripts must describe the same utterance. Later
                # requests can upload now, but cannot advance LiveKit's turn state.
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        stt.SpeechEventType.START_OF_SPEECH,
                        speech_start_time=utterance.speech_start_time,
                    )
                )
                utterance.publishing = True
                for event in (utterance.interim, utterance.final, utterance.usage):
                    if event is not None:
                        self._event_ch.send_nowait(event)
                utterance.interim = utterance.final = utterance.usage = None
                await utterance.done.wait()
                # Closing the audio source is not a transcript completion signal.
                # Publish the end only after the final, including buffered results.
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        stt.SpeechEventType.END_OF_SPEECH,
                        speech_end_time=utterance.speech_end_time,
                    )
                )
                pending.remove(utterance)

        def upload(utterance: _Utterance, frames: list[rtc.AudioFrame]) -> None:
            nonlocal pending_bytes
            for frame in frames:
                if frame.num_channels != 1 or frame.sample_rate != 16000:
                    raise APIError(
                        "Rime streaming requires mono 16 kHz VAD frames", retryable=False
                    )
                data = bytes(frame.data)
                # Bound all pending uploads, not just the newest utterance.
                if pending_bytes + len(data) > 320000:
                    raise APIError(
                        "Rime streaming audio backlog exceeded 10 seconds", retryable=False
                    )
                pending_bytes += len(data)
                utterance.audio_duration += frame.duration
                utterance.audio.put_nowait(data)

        def finish(utterance: _Utterance) -> None:
            if utterance.speech_end_time is None:
                utterance.speech_end_time = time.time()
            utterance.audio.put_nowait(None)

        async def segment() -> bool:
            detector = self._vad.stream()
            eof = False

            async def forward() -> None:
                nonlocal eof
                async for frame in self._input_ch:
                    if isinstance(frame, self._FlushSentinel):
                        break
                    if frame.num_channels != 1:
                        raise APIError("Rime streaming requires mono audio", retryable=False)
                    detector.push_frame(frame)
                else:
                    eof = True
                detector.end_input()

            async def recognize() -> None:
                active: _Utterance | None = None
                async for event in detector:
                    if event.type == vad.VADEventType.START_OF_SPEECH:
                        if active is not None:
                            raise APIError("Overlapping Rime VAD utterances", retryable=False)
                        if len(pending) >= 8:
                            raise APIError(
                                "Rime streaming pending utterance limit exceeded", retryable=False
                            )
                        owner = self._owner
                        options: dict[str, Any] = {
                            "language": self._language
                            if utils.is_given(self._language)
                            else owner._language,
                            "mode": owner._mode,
                            "context_terms": tuple(
                                dict.fromkeys((*owner._context_terms, *owner._session_keyterms))
                            ),
                            "timeout": self._conn_options.timeout,
                        }
                        contract_type = getattr(owner._sdk, "TranscriptionOutputContract", None)
                        if contract_type is not None:
                            options["output_contract"] = contract_type(owner._output_contract)
                        active = _Utterance()
                        upload(active, event.frames)
                        pending.add(active)
                        ordered.put_nowait(active)
                        task = asyncio.create_task(transcribe(active, options))
                        requests.add(task)
                        task.add_done_callback(finished)
                    elif event.type == vad.VADEventType.INFERENCE_DONE and active is not None:
                        upload(active, event.frames)
                    elif event.type == vad.VADEventType.END_OF_SPEECH and active is not None:
                        # END contains the whole utterance; don't upload it again.
                        active.speech_end_time = (
                            time.time() - event.silence_duration - event.inference_duration
                        )
                        finish(active)
                        active = None
                if active is not None:
                    finish(active)

            tasks = [asyncio.create_task(forward()), asyncio.create_task(recognize())]
            try:
                await asyncio.gather(*tasks)
                return eof
            finally:
                await utils.aio.cancel_and_wait(*tasks)
                await detector.aclose()

        async def read_audio() -> None:
            while not await segment():
                pass
            ordered.put_nowait(None)

        tasks = [asyncio.create_task(read_audio()), asyncio.create_task(emit_results())]
        main = asyncio.gather(*tasks)
        try:
            done, _ = await asyncio.wait((main, failure), return_when=asyncio.FIRST_COMPLETED)
            if failure in done:
                raise failure.result()
            await main
        finally:
            main.cancel()
            await utils.aio.cancel_and_wait(*tasks, *tuple(requests))
            await asyncio.gather(main, return_exceptions=True)
            failure.cancel()

    async def aclose(self) -> None:
        try:
            await super().aclose()
        finally:
            self._owner._streams.discard(self)
