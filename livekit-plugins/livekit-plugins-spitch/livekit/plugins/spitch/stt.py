from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import httpx

import spitch
from livekit import rtc
from livekit.agents import (
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    Language,
    NotGivenOr,
)
from livekit.agents.stt import stt
from livekit.agents.utils import AudioBuffer
from livekit.agents.voice.io import TimedString
from spitch import AsyncSpitch


@dataclass
class _STTOptions:
    language: Language


class STT(stt.STT):
    def __init__(self, *, language: str = "en") -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                # word timestamps don't seem to work despite the docs saying they do
                aligned_transcript=False,
            )
        )

        self._opts = _STTOptions(language=Language(language))
        self._client = AsyncSpitch()

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "Spitch"

    def update_options(self, language: str) -> None:
        self._opts.language = Language(language) if language else self._opts.language

    def _sanitize_options(self, *, language: str | None = None) -> _STTOptions:
        config = dataclasses.replace(self._opts)
        if language:
            config.language = Language(language)
        return config

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            config = self._sanitize_options(language=language or None)
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()
            model = "mansa_v1" if config.language == "en" else "legacy"
            resp = await self._client.speech.transcribe(
                language=config.language.language,  # type: ignore
                content=data,
                timeout=httpx.Timeout(30, connect=conn_options.timeout),
                timestamp="word" if "mansa" in model else None,
            )

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=resp.text or "",
                        language=Language(config.language or ""),
                        start_time=float(resp.segments[0].start)
                        if resp.segments and resp.segments[0] and resp.segments[0].start
                        else 0.0,
                        end_time=float(resp.segments[-1].end)
                        if resp.segments and resp.segments[-1] and resp.segments[-1].end
                        else 0.0,
                        words=[
                            TimedString(
                                text=str(segment.text) if segment.text else "",
                                start_time=float(segment.start) if segment.start else 0.0,
                                end_time=float(segment.end) if segment.end else 0.0,
                            )
                            for segment in resp.segments
                            if segment is not None
                        ]
                        if resp.segments
                        else None,
                    ),
                ],
            )
        except spitch.APITimeoutError as e:
            raise APITimeoutError() from e
        except spitch.APIStatusError as e:
            raise APIStatusError(e.message, status_code=e.status_code, body=e.body) from e
        except Exception as e:
            raise APIConnectionError() from e
