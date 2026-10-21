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
from typing import TYPE_CHECKING, Any

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, APIStatusError, stt, utils, vad
from livekit.agents.language import LanguageCode
from livekit.agents.types import NotGivenOr

if TYPE_CHECKING:
    from .stt import STT


class SpeechStream(stt.RecognizeStream):
    """One SDK call per VAD utterance; never replay partially consumed live audio."""

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
        while not await self._run_segment():
            pass

    async def _run_segment(self) -> bool:
        detector = self._vad.stream()
        eof = False
        failure: asyncio.Future[BaseException] = asyncio.get_running_loop().create_future()
        recognition: asyncio.Task[None] | None = None
        queue: asyncio.Queue[bytes | None] | None = None
        pending_bytes = 0
        audio_duration = 0.0
        speech_end_time: float | None = None

        async def audio_source(source: asyncio.Queue[bytes | None]) -> AsyncIterator[bytes]:
            nonlocal pending_bytes
            while (chunk := await source.get()) is not None:
                pending_bytes -= len(chunk)
                yield chunk

        async def transcribe(source: asyncio.Queue[bytes | None], options: dict[str, Any]) -> None:
            text = ""
            try:
                async with self._owner._client.stt.stream(
                    audio_source(source), **options
                ) as stream:
                    async for update in stream:
                        # LiveKit interims are replacement snapshots, even for permanent deltas.
                        kind = str(update.kind)
                        if kind == "delta":
                            text += update.text
                        else:
                            text = update.text
                        final = kind == "final"
                        self._event_ch.send_nowait(
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
                                speech_end_time=speech_end_time if final else None,
                            )
                        )
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.RECOGNITION_USAGE,
                            request_id=stream.request_id or "",
                            recognition_usage=stt.RecognitionUsage(audio_duration=audio_duration),
                        )
                    )
            except self._owner._sdk.RimeError as exc:
                raise APIStatusError(
                    "Rime streaming STT failed",
                    request_id=exc.request_id,
                    retryable=False,
                ) from None

        def failed(task: asyncio.Task[None]) -> None:
            if (
                not task.cancelled()
                and (exc := task.exception()) is not None
                and not failure.done()
            ):
                failure.set_result(exc)

        def upload(frames: list[rtc.AudioFrame]) -> None:
            nonlocal pending_bytes, audio_duration
            assert queue is not None
            for frame in frames:
                if frame.num_channels != 1 or frame.sample_rate != 16000:
                    raise APIError(
                        "Rime streaming requires mono 16 kHz VAD frames", retryable=False
                    )
                data = bytes(frame.data)
                # Ten seconds of PCM16, independent of the detector's frame size.
                if pending_bytes + len(data) > 320000:
                    raise APIError(
                        "Rime streaming audio backlog exceeded 10 seconds", retryable=False
                    )
                pending_bytes += len(data)
                audio_duration += frame.duration
                queue.put_nowait(data)

        async def finish() -> None:
            nonlocal queue, recognition, speech_end_time
            if queue is None:
                return
            if speech_end_time is None:
                speech_end_time = time.time()
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    stt.SpeechEventType.END_OF_SPEECH,
                    speech_end_time=speech_end_time,
                )
            )
            queue.put_nowait(None)
            assert recognition is not None
            await recognition
            queue = None
            recognition = None

        async def forward() -> None:
            nonlocal eof
            async for frame in self._input_ch:
                if isinstance(frame, self._FlushSentinel):
                    # Drain and restart VAD at an explicit segment boundary. Some VADs
                    # reset silently on flush without emitting END_OF_SPEECH.
                    break
                if frame.num_channels != 1:
                    raise APIError("Rime streaming requires mono audio", retryable=False)
                detector.push_frame(frame)
            else:
                eof = True
            detector.end_input()

        async def recognize() -> None:
            nonlocal queue, recognition, speech_end_time, audio_duration
            async for event in detector:
                if event.type == vad.VADEventType.START_OF_SPEECH:
                    if queue is not None:
                        raise APIError("Overlapping Rime VAD utterances", retryable=False)
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
                    speech_end_time = None
                    audio_duration = 0.0
                    queue = asyncio.Queue()
                    upload(event.frames)
                    self._event_ch.send_nowait(stt.SpeechEvent(stt.SpeechEventType.START_OF_SPEECH))
                    recognition = asyncio.create_task(transcribe(queue, options))
                    recognition.add_done_callback(failed)
                elif event.type == vad.VADEventType.INFERENCE_DONE and queue is not None:
                    upload(event.frames)
                elif event.type == vad.VADEventType.END_OF_SPEECH and queue is not None:
                    # END contains the whole utterance; it must not be uploaded again.
                    speech_end_time = (
                        time.time() - event.silence_duration - event.inference_duration
                    )
                    await finish()
            await finish()

        tasks = [asyncio.create_task(forward()), asyncio.create_task(recognize())]
        main = asyncio.gather(*tasks)
        try:
            done, _ = await asyncio.wait((main, failure), return_when=asyncio.FIRST_COMPLETED)
            if failure in done:
                raise failure.result()
            await main
            return eof
        finally:
            # gather does not cancel siblings when one raises.
            main.cancel()
            await utils.aio.cancel_and_wait(*tasks)
            await asyncio.gather(main, return_exceptions=True)
            if recognition is not None:
                await utils.aio.cancel_and_wait(recognition)
            failure.cancel()
            await detector.aclose()

    async def aclose(self) -> None:
        try:
            await super().aclose()
        finally:
            self._owner._streams.discard(self)
