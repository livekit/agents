# Copyright 2025 LiveKit, Inc.
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

"""Speech-to-Text implementation for ConvoZen Akshara.

Akshara is a batch recognizer: it takes a complete utterance and returns one
transcript, with no interim results. It is therefore declared as a non-streaming
STT, and ``Agent.stt_node`` wraps it in :class:`livekit.agents.stt.StreamAdapter`
using the session's VAD to cut the audio into utterances. A VAD (for example
``silero.VAD.load()``) is required on the ``AgentSession``.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import is_given

from .log import logger
from .models import DEFAULT_BASE_URL, STT_LANGUAGES, STTLanguages, STTModels

if TYPE_CHECKING:
    from livekit.agents.utils import AudioBuffer

_TRANSCRIBE_PATH = "/v2/akshara/transcribe"

# Only so much of a server error body is worth carrying into an exception message.
_MAX_ERROR_BODY = 512


@dataclass
class _STTOptions:
    api_key: str
    base_url: str
    model: STTModels | str
    language: str
    lang_tags: list[str] | None
    keywords: list[str]
    blank_penalty: float | None
    word_timestamps: bool
    # Keyterms pushed by the framework (session config + auto-detection), kept
    # apart from the user's own `keywords` so `update_options` can replace one
    # without clobbering the other.
    session_keyterms: list[str] = field(default_factory=list)

    def transcribe_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{_TRANSCRIBE_PATH}"

    def all_keywords(self) -> list[str]:
        """User keywords merged with framework-supplied keyterms, order preserved."""
        return list(dict.fromkeys([*self.keywords, *self.session_keyterms]))


def _resolve_lang_tags(language: str, lang_tags: NotGivenOr[list[str] | None]) -> list[str] | None:
    """Work out which `lang_tags` to send, validating against the server's set.

    An explicit ``lang_tags`` wins and is validated strictly. Otherwise the tag is
    derived from ``language``, and only when that language is one Akshara accepts as
    a tag — an unrecognized ``language`` sends no hint rather than a request the
    server would reject outright.
    """
    if is_given(lang_tags):
        if lang_tags is None:
            return None
        invalid = set(lang_tags) - STT_LANGUAGES
        if invalid:
            raise ValueError(
                f"invalid lang_tags: {sorted(invalid)}. valid tags: {sorted(STT_LANGUAGES)}"
            )
        return list(lang_tags)

    return [language] if language in STT_LANGUAGES else None


class STT(stt.STT):
    """ConvoZen Akshara speech-to-text.

    Akshara covers nine Indian languages plus code-mixed speech.
    """

    def __init__(
        self,
        *,
        language: STTLanguages | str = "en",
        model: STTModels | str = "akshara-pro",
        lang_tags: NotGivenOr[list[str] | None] = NOT_GIVEN,
        keywords: list[str] | None = None,
        blank_penalty: float | None = None,
        word_timestamps: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of ConvoZen Akshara STT.

        Args:
            language: Language of the audio, reported back on every transcript. Also
                supplies the default ``lang_tags`` hint when it is one Akshara accepts.
            model: ``"akshara-pro"`` (default) or the base ``"akshara"``.
            lang_tags: Explicit language hints, e.g. ``["hi", "en"]`` for Hindi/English
                code-mixing. Overrides the tag derived from ``language``; pass ``None``
                to send no hint at all.
            keywords: Vocabulary boosting list — names, jargon, product terms.
            blank_penalty: Penalizes blank/silence tokens; increase it to reduce empty
                gaps in the transcript. Unbounded — it is a raw penalty weight, not a
                normalized ``[0, 1]`` value. ``None`` lets the server derive one from
                ``lang_tags``.
            word_timestamps: Request per-word timings, surfaced as
                ``SpeechData.words``. Timings are relative to the start of each
                recognized utterance rather than the audio stream, so the STT does
                not declare ``aligned_transcript``. Adds a little server-side work.
            api_key: ConvoZen API key. Falls back to the ``CONVOZEN_API_KEY``
                environment variable.
            base_url: API base URL. Falls back to ``CONVOZEN_BASE_URL``, then to the
                public endpoint.
            http_session: An existing aiohttp session to use. By default the shared
                LiveKit session is used.

        Raises:
            ValueError: If no API key is available, or ``lang_tags`` contains a tag
                Akshara does not accept.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                keyterms=True,
                # Word timings are relative to each recognized utterance, not the
                # audio stream, so they don't meet the aligned-transcript contract.
                aligned_transcript=False,
            )
        )

        convozen_api_key = api_key or os.environ.get("CONVOZEN_API_KEY")
        if not convozen_api_key:
            raise ValueError(
                "ConvoZen API key is required. "
                "Provide it directly or set the CONVOZEN_API_KEY environment variable."
            )

        self._opts = _STTOptions(
            api_key=convozen_api_key,
            base_url=base_url or os.environ.get("CONVOZEN_BASE_URL") or DEFAULT_BASE_URL,
            model=model,
            language=language,
            lang_tags=_resolve_lang_tags(language, lang_tags),
            keywords=list(keywords) if keywords else [],
            blank_penalty=blank_penalty,
            word_timestamps=word_timestamps,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        return "ConvoZen"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        lang_tags: NotGivenOr[list[str] | None] = NOT_GIVEN,
        keywords: NotGivenOr[list[str]] = NOT_GIVEN,
        blank_penalty: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        """Update recognition options for subsequent requests."""
        if is_given(model):
            self._opts.model = model
        if is_given(keywords):
            self._opts.keywords = list(keywords)
        if is_given(blank_penalty):
            self._opts.blank_penalty = blank_penalty
        if is_given(language):
            self._opts.language = language
        if is_given(language) or is_given(lang_tags):
            # Re-derive so that changing the language alone also moves the hint,
            # and so an explicit tag list is validated the same way as at __init__.
            self._opts.lang_tags = _resolve_lang_tags(
                self._opts.language,
                lang_tags if is_given(lang_tags) else NOT_GIVEN,
            )

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        # Framework-managed keyterms map onto Akshara's `keywords` vocabulary boost.
        self._opts.session_keyterms = list(keyterms)

    def _build_form(self, wav_bytes: bytes, opts: _STTOptions) -> aiohttp.FormData:
        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", str(opts.model))
        # LiveKit hands us a single mixed track, so recognition is always mono.
        # Diarization and the stereo/two-speaker path are deliberately not exposed:
        # they return a different response shape and mean nothing for one agent track.
        form.add_field("audio_channels", "mono")

        if opts.lang_tags:
            form.add_field("lang_tags", json.dumps(opts.lang_tags))
        if keywords := opts.all_keywords():
            form.add_field("keywords", json.dumps(keywords))
        if opts.blank_penalty is not None:
            form.add_field("blank_penalty", str(opts.blank_penalty))
        if opts.word_timestamps:
            form.add_field("word_timestamps", "true")

        return form

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        opts = replace(self._opts)
        if is_given(language) and language != opts.language:
            opts.language = language
            opts.lang_tags = _resolve_lang_tags(language, NOT_GIVEN)

        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        try:
            async with self._ensure_session().post(
                url=opts.transcribe_url(),
                data=self._build_form(wav_bytes, opts),
                headers={"x-api-key": opts.api_key},
                timeout=aiohttp.ClientTimeout(
                    total=conn_options.timeout,
                    sock_connect=conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    body = (await res.text())[:_MAX_ERROR_BODY]
                    raise APIStatusError(
                        message=f"ConvoZen Akshara returned {res.status}: {body}",
                        status_code=res.status,
                        request_id=None,
                        body=body,
                    )

                data = await res.json()

        except APIStatusError:
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            # `from None`, not `from e`: aiohttp embeds the RequestInfo — URL and
            # request headers, which carry the API key — in its exception chain.
            raise APIConnectionError(type(e).__name__) from None

        return self._to_speech_event(data, opts)

    def _to_speech_event(self, data: dict, opts: _STTOptions) -> stt.SpeechEvent:
        text = data.get("text") or ""

        words: list[TimedString] | None = None
        if raw_words := data.get("word_timestamps"):
            try:
                words = [
                    TimedString(
                        text=w["word"],
                        start_time=w["start_s"],
                        end_time=w["end_s"],
                    )
                    for w in raw_words
                ]
            except (KeyError, TypeError):
                logger.warning("could not parse Akshara word_timestamps, dropping them")

        speech_data = stt.SpeechData(
            language=LanguageCode(opts.language),
            text=text,
            words=words,
            # `score` is a log-probability (e.g. -2.45, closer to 0 is better), not a
            # [0, 1] confidence. Assigning it to `confidence` would make a good
            # transcript look near-zero to anything that thresholds on that field, so
            # it is passed through as metadata and `confidence` is left at its default.
            metadata={"score": score} if (score := data.get("score")) is not None else None,
        )
        if words:
            speech_data.start_time = words[0].start_time or 0.0
            speech_data.end_time = words[-1].end_time or 0.0

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid(),
            alternatives=[speech_data],
        )
