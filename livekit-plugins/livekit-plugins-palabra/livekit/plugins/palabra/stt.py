# Copyright 2023 LiveKit, Inc.
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

"""Palabra realtime speech-to-text plugin for LiveKit Agents.

Transport is the official ``palabra-ai`` SDK (``palabra_ai.SttSession``).
This module is the LiveKit adapter on top of it.
With ``translate_languages`` set, each emitted ``SpeechData`` maps as:
``language``/``text`` = the translation; ``source_languages``/``source_texts`` = the original.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import weakref
from dataclasses import dataclass, field

from palabra_ai import (
    AuthError,
    Palabra,
    PalabraError,
    ServerError,
    SessionError,
    SttSession,
    SttTranscript,
    TaskError,
)

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

# ASR recommended input rate; the base RecognizeStream resamples room audio to it.
DEFAULT_STT_SAMPLE_RATE = 16000

# Room audio arrives as 10 ms frames; re-buffer into 100 ms binary WS frames.
_CHUNK_MS = 100

# Server `error` codes that are worth retrying.
_RETRYABLE_ERROR_CODES = frozenset({"SERVICE_UNAVAILABLE", "RATE_LIMIT_EXCEEDED"})

# Finished utterances stay in the registry (size-capped).
# A late event for a finished transcription_id must be ignored, not restart the utterance.
_MAX_TRACKED_UTTERANCES = 128


@dataclass
class _STTOptions:
    language: str | None  # None: server auto-detects the spoken language
    translate_languages: list[str]
    filler_filter: bool | None  # None: the server default is used
    sample_rate: int


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str | None = None,
        translate_languages: str | list[str] | None = None,
        filler_filter: bool | None = None,
        sample_rate: int = DEFAULT_STT_SAMPLE_RATE,
        region: str | None = None,
    ) -> None:
        """Create a new instance of Palabra realtime STT.

        Args:
            api_key: Palabra API Key (create one at https://platform.palabra.ai/api-keys).
                Falls back to the ``PALABRA_API_KEY`` environment variable.
            language: Source language code. Defaults to ``None`` (server auto-detect).
            translate_languages: Target language(s) for live translation. When set, the
                emitted ``SpeechData`` maps as ``language``/``text`` = the translation and
                ``source_languages``/``source_texts`` = the original speech.
                When several target languages are requested, only the first translation
                to finalize is emitted; multi-target aggregation is planned.
            filler_filter: Drop filler words ("uh", "um") from transcripts.
                Unset by default, so the server default is used.
            sample_rate: Input sample rate in Hz sent to the server; room audio is
                resampled to it. Defaults to ``16000`` (the recommended ASR rate).
            region: Palabra region (``"eu"`` / ``"us"``). Falls back to the
                ``PALABRA_REGION`` environment variable, then ``"eu"``.

        Raises:
            ValueError: no ``api_key`` argument and no ``PALABRA_API_KEY`` in the environment.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=False,
            )
        )
        api_key = api_key or os.environ.get("PALABRA_API_KEY")
        if not api_key:
            raise ValueError(
                "Palabra api_key is required, either as an argument or set the"
                " PALABRA_API_KEY environment variable"
            )

        if translate_languages is None:
            translate_languages = []
        elif isinstance(translate_languages, str):
            translate_languages = [translate_languages]
        else:
            translate_languages = list(translate_languages)

        self._client = Palabra(api_key, region=region)
        self._opts = _STTOptions(
            language=language,
            translate_languages=translate_languages,
            filler_filter=filler_filter,
            sample_rate=sample_rate,
        )
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        """Recognition model id; always ``"default"`` — the server picks the model."""
        return "default"

    @property
    def provider(self) -> str:
        """Provider display name used in metrics."""
        return "Palabra"

    def update_options(
        self,
        *,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        translate_languages: NotGivenOr[str | list[str] | None] = NOT_GIVEN,
        filler_filter: NotGivenOr[bool | None] = NOT_GIVEN,
    ) -> None:
        """Update recognition options. Changes apply to new streams only."""
        if is_given(language):
            self._opts.language = language
        if is_given(translate_languages):
            if translate_languages is None:
                self._opts.translate_languages = []
            elif isinstance(translate_languages, str):
                self._opts.translate_languages = [translate_languages]
            else:
                self._opts.translate_languages = list(translate_languages)
        if is_given(filler_filter):
            self._opts.filler_filter = filler_filter

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "Palabra STT does not support batch recognition — use streaming via stream()"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Open a streaming recognition session; ``language`` overrides the default."""
        opts = _STTOptions(
            language=language if is_given(language) else self._opts.language,
            translate_languages=list(self._opts.translate_languages),
            filler_filter=self._opts.filler_filter,
            sample_rate=self._opts.sample_rate,
        )
        stream = SpeechStream(stt=self, opts=opts, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close all active recognition streams."""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


@dataclass
class _Utterance:
    """Emission state of one server-side transcription (keyed by transcription_id)."""

    started: bool = False  # START_OF_SPEECH has been emitted
    source: SttTranscript | None = None  # latest source-language transcript (translation mode)
    done: bool = field(default=False)  # FINAL_TRANSCRIPT has been emitted


class SpeechStream(stt.RecognizeStream):
    """Streaming speech recognition over one Palabra STT WebSocket session.

    Created by ``STT.stream()``; not instantiated directly.
    """

    def __init__(self, *, stt: STT, opts: _STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._stt: STT = stt
        self._opts = opts

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        input_closed = asyncio.Event()
        utterances: dict[str, _Utterance] = {}

        def _make_session() -> SttSession:
            return self._stt._client.stt(
                self._opts.language,
                sample_rate=self._opts.sample_rate,
                translate_languages=self._opts.translate_languages or None,
                enable_filler_filter=self._opts.filler_filter,
            )

        async def _send_task(session: SttSession) -> None:
            samples_per_chunk = self._opts.sample_rate * _CHUNK_MS // 1000
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_chunk,
            )
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.push(data.data.tobytes())
                for frame in frames:
                    await session.send_audio(frame.data.tobytes())
            for frame in audio_bstream.flush():
                await session.send_audio(frame.data.tobytes())
            input_closed.set()
            # The protocol has no end-of-stream message; closing the WS is the signal.
            # The close also makes session.receive() return None, which ends _recv_task.
            await session.close()

        def _speech_data(ev: SttTranscript) -> stt.SpeechData:
            offset = self.start_time_offset
            data = stt.SpeechData(
                language=LanguageCode(ev.language),
                text=ev.text,
                start_time=(ev.start_time or 0.0) + offset,
                end_time=(ev.end_time or 0.0) + offset,
            )
            if ev.is_translation:
                source = utterances[ev.transcription_id].source
                if source is not None:
                    data.source_languages = [LanguageCode(source.language)]
                    data.source_texts = [source.text]
                data.target_languages = [LanguageCode(ev.language)]
                data.target_texts = [ev.text]
            return data

        def _emit(ev: SttTranscript) -> None:
            utt = utterances.setdefault(ev.transcription_id, _Utterance())
            if utt.done:
                return

            translation_active = bool(self._opts.translate_languages)
            if translation_active and not ev.is_translation:
                # Source-language event: stored for pairing; only the translation is emitted.
                utt.source = ev
                return
            if not ev.text and not utt.started:
                return  # filler-filter can produce empty segments; nothing to report

            if not utt.started:
                utt.started = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH, request_id=request_id)
                )
            if not ev.is_eos:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=request_id,
                        alternatives=[_speech_data(ev)],
                    )
                )
                return

            utt.done = True
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[_speech_data(ev)],
                )
            )
            self._event_ch.send_nowait(
                stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, request_id=request_id)
            )
            # Keep the finished entry on purpose.
            # The server can resend this id: segment updates, extra translation targets.
            # The `done` check above drops those events.
            utt.source = None  # the stored source transcript is no longer needed
            if len(utterances) > _MAX_TRACKED_UTTERANCES:
                for tid, u in list(utterances.items()):  # insertion order = oldest first
                    if not u.done:
                        continue
                    utterances.pop(tid, None)
                    if len(utterances) <= _MAX_TRACKED_UTTERANCES:
                        break

        async def _recv_task(session: SttSession) -> None:
            async for event in session:
                if isinstance(event, ServerError):
                    logger.error("palabra stt error: %s", event.code)
                    raise APIError(
                        f"palabra stt error: {event.code}",
                        retryable=event.code in _RETRYABLE_ERROR_CODES,
                    )
                if isinstance(event, SttTranscript):
                    _emit(event)
            # The iterator ends when the connection closes.
            # That is expected only after _send_task closed the WS at end of input.
            if not input_closed.is_set():
                raise APIConnectionError("palabra stt connection closed unexpectedly")

        session: SttSession | None = None
        try:
            # The session object exists before connecting, so the finally below
            # also closes a connection abandoned by the timeout.
            session = _make_session()
            await asyncio.wait_for(session.__aenter__(), timeout=self._conn_options.timeout)
            tasks = [
                asyncio.create_task(_send_task(session)),
                asyncio.create_task(_recv_task(session)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except AuthError as e:
            raise APIStatusError(str(e), status_code=401, request_id=None, body=None) from None
        except TaskError as e:
            # `code` is a short server enum (e.g. VALIDATION_ERROR); `desc` is free-form
            # server text, so only the code is surfaced.
            raise APIConnectionError(f"palabra stt error: {e.code}") from None
        except (SessionError, PalabraError) as e:
            # The SDK interpolates the underlying websockets error into its message, and the
            # connect URL carries the API key as a ?token= query parameter -- so neither
            # str(e) nor the __cause__ chain is safe to surface.
            raise APIConnectionError(type(e).__name__) from None
        finally:
            if session is not None:
                with contextlib.suppress(Exception):
                    await session.close()
