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
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from livekit.agents import APIConnectOptions, APIError, APIStatusError, stt, utils, vad
from livekit.agents.language import LanguageCode
from livekit.agents.types import NotGivenOr

if TYPE_CHECKING:
    from .stt import STT


class SpeechStream(stt.RecognizeStream):
    """Recognize one utterance at a time, sending its audio as the VAD reports it.

    The SDK reads each utterance's audio from the VAD's events, so no audio is
    queued here. An utterance detected before the previous final transcript
    arrives waits in the VAD's event stream.

    Speech bounds are the VAD's own timestamps. A VAD stream counts from its first
    frame, and a flush resets that count without saying when it took effect. Each
    flush therefore ends the VAD stream, and the next one starts with the stream's
    connection clock, after every event of the flushed segment is emitted.
    """

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
        while not await self._recognize_segment():
            # The next frame read is the next VAD stream's first.
            self._restart_connection_clock()

    async def _recognize_segment(self) -> bool:
        """Recognize the input up to the next flush. Return whether the input ended."""
        detector = self._vad.stream()
        input_ended = False

        async def forward() -> None:
            nonlocal input_ended
            async for frame in self._input_ch:
                if isinstance(frame, self._FlushSentinel):
                    break
                if frame.num_channels != 1:
                    raise APIError("Rime streaming requires mono audio", retryable=False)
                detector.push_frame(frame)
            else:
                input_ended = True
            detector.end_input()

        async def recognize() -> None:
            try:
                events = aiter(detector)
                async for event in events:
                    if event.type == vad.VADEventType.START_OF_SPEECH:
                        await self._recognize_utterance(event, events)
            finally:
                forward_task.cancel()

        forward_task = asyncio.create_task(forward())
        tasks = [forward_task, asyncio.create_task(recognize())]
        try:
            await asyncio.gather(*tasks)
            return input_ended
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await detector.aclose()

    async def _recognize_utterance(
        self, start: vad.VADEvent, events: AsyncIterator[vad.VADEvent]
    ) -> None:
        """Transcribe the utterance ``start`` opens, reading ``events`` through its end."""
        owner = self._owner
        options: dict[str, Any] = {
            "language": self._language if utils.is_given(self._language) else owner._language,
            "mode": owner._mode,
            "context_terms": tuple(
                dict.fromkeys((*owner._context_terms, *owner._session_keyterms))
            ),
            "timeout": self._conn_options.timeout,
        }
        contract_type = getattr(owner._sdk, "TranscriptionOutputContract", None)
        if contract_type is not None:
            options["output_contract"] = contract_type(owner._output_contract)

        speech_end_time: float | None = None
        audio_duration = 0.0
        audio_error: APIError | None = None

        def pcm(event: vad.VADEvent) -> bytes:
            nonlocal audio_duration
            for frame in event.frames:
                if frame.num_channels != 1 or frame.sample_rate != 16000:
                    raise APIError(
                        "Rime streaming requires mono 16 kHz VAD frames", retryable=False
                    )
                audio_duration += frame.duration
            return b"".join(bytes(frame.data) for frame in event.frames)

        async def audio() -> AsyncIterator[bytes]:
            nonlocal speech_end_time, audio_error
            try:
                last = start
                if data := pcm(start):
                    yield data
                async for event in events:
                    last = event
                    if event.type == vad.VADEventType.END_OF_SPEECH:
                        # END contains the whole utterance; don't upload it again.
                        speech_end_time = event.timestamp - event.silence_duration
                        return
                    if event.type == vad.VADEventType.INFERENCE_DONE and (data := pcm(event)):
                        yield data
                # A flush ended the detector mid-speech, at the last audio it reported.
                speech_end_time = last.timestamp
            except APIError as error:
                audio_error = error  # the SDK reports a failed source as its own error
                raise

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                stt.SpeechEventType.START_OF_SPEECH,
                speech_start_time=start.timestamp - start.speech_duration,
            )
        )
        text = ""
        try:
            async with owner._client.stt.stream(audio(), **options) as stream:
                async for update in stream:
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
                            # The final follows the end of the audio.
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
        except owner._sdk.RimeError as exc:
            if audio_error is not None:
                raise audio_error from None
            raise APIStatusError(
                "Rime streaming STT failed",
                request_id=exc.request_id,
                retryable=False,
            ) from None
        # Closing the audio source is not a transcript completion signal.
        # Publish the end only after the final.
        self._event_ch.send_nowait(
            stt.SpeechEvent(stt.SpeechEventType.END_OF_SPEECH, speech_end_time=speech_end_time)
        )

    async def aclose(self) -> None:
        try:
            await super().aclose()
        finally:
            self._owner._streams.discard(self)
