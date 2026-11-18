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


from __future__ import annotations

import dataclasses
import json
import os
import time
import weakref
from dataclasses import dataclass
from typing import ClassVar, Literal, NotRequired, TypedDict, assert_never, final
from urllib.parse import urlencode

from pydantic import JsonValue, TypeAdapter, ValidationError

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    LanguageCode,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TranscriptChunk

from .log import logger


class _Word(TypedDict):
    text: str
    start: float
    end: float
    confidence: float


@final
class _Begin(TypedDict):
    type: Literal["Begin"]
    id: str
    expires_at: int


@final
class _SpeechStarted(TypedDict):
    type: Literal["SpeechStarted"]
    timestamp: NotRequired[float]


@final
class _Turn(TypedDict):
    type: Literal["Turn"]
    words: list[_Word]
    end_of_turn: NotRequired[bool]
    end_of_turn_confidence: NotRequired[float]
    turn_is_formatted: NotRequired[bool]
    utterance: NotRequired[str]
    transcript: NotRequired[str]
    language_code: NotRequired[str]
    language_confidence: NotRequired[float]
    speaker_label: NotRequired[str]


@final
class _Termination(TypedDict):
    type: Literal["Termination"]


@final
class _Error(TypedDict):
    """The error AssemblyAI sends, untagged, before it closes with 3005."""

    type: NotRequired[Literal[None]]
    error: str


_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


class _Out(TypedDict):
    type: Literal["Terminate", "ForceEndpoint", "UpdateConfiguration"]
    prompt: NotRequired[str]
    agent_context: NotRequired[str]
    keyterms_prompt: NotRequired[list[str]]
    language_codes: NotRequired[list[str]]
    max_turn_silence: NotRequired[int]
    min_turn_silence: NotRequired[int]
    end_of_turn_confidence_threshold: NotRequired[float]
    continuous_partials: NotRequired[bool]
    interruption_delay: NotRequired[int]
    vad_threshold: NotRequired[float]


@dataclass
class STTOptions:
    sample_rate: int
    buffer_size_seconds: float
    encoding: Literal["pcm_s16le", "pcm_mulaw"] = "pcm_s16le"
    speech_model: Literal[
        "universal-streaming-english",
        "universal-streaming-multilingual",
        "u3-rt-pro",
        "u3-rt-pro-beta-1",
        "universal-3-5-pro",
        "universal-3-6-pro",
    ] = "universal-3-6-pro"
    language_detection: NotGivenOr[bool] = NOT_GIVEN
    language_codes: NotGivenOr[list[str]] = NOT_GIVEN
    end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN
    min_turn_silence: NotGivenOr[int] = NOT_GIVEN
    max_turn_silence: NotGivenOr[int] = NOT_GIVEN
    format_turns: NotGivenOr[bool] = NOT_GIVEN
    continuous_partials: NotGivenOr[bool] = NOT_GIVEN
    interruption_delay: NotGivenOr[int] = NOT_GIVEN
    keyterms_prompt: NotGivenOr[list[str]] = NOT_GIVEN
    prompt: NotGivenOr[str] = NOT_GIVEN
    agent_context: NotGivenOr[str] = NOT_GIVEN
    previous_context_n_turns: NotGivenOr[int] = NOT_GIVEN
    vad_threshold: NotGivenOr[float] = NOT_GIVEN
    speaker_labels: NotGivenOr[bool] = NOT_GIVEN
    max_speakers: NotGivenOr[int] = NOT_GIVEN
    domain: NotGivenOr[str] = NOT_GIVEN
    voice_focus: NotGivenOr[Literal["near-field", "far-field"]] = NOT_GIVEN
    voice_focus_threshold: NotGivenOr[float] = NOT_GIVEN
    mode: NotGivenOr[Literal["min_latency", "balanced", "max_accuracy"]] = NOT_GIVEN


# Speech models in the Universal-3 Pro family, which share the same parameter support
# (prompt, agent_context, previous_context_n_turns, continuous_partials,
# interruption_delay, voice_focus, voice_focus_threshold) and connect-time
# defaults. Mirrors the server-side `SpeechModel.is_u3_pro`.
_U3_PRO_MODELS = ("u3-rt-pro", "u3-rt-pro-beta-1", "universal-3-5-pro", "universal-3-6-pro")

# Server-side cap on the number of steering codes, mirrored client-side so bad
# input fails at construction/update time instead of as a websocket error.
_MAX_LANGUAGE_CODES = 10


def _normalize_language_codes(language_codes: str | list[str]) -> list[str]:
    """Normalize code(s) to bare ISO 639-1 and dedup preserving order.

    Accepts a single code or a list, mirroring the server, which takes either.
    Mirrors the AssemblyAI streaming API's validation rules: at most 10 codes,
    and 'multi' (the unsteered multilingual default) must be sent alone.
    """
    if isinstance(language_codes, str):
        # A bare code is shorthand for a one-element list; wrapping here (the
        # single choke point for every input path) keeps a stray string from
        # being iterated per-character. "" means "no codes", like [].
        language_codes = [language_codes] if language_codes else []
    normalized = list(dict.fromkeys(LanguageCode(code).language for code in language_codes))
    if len(normalized) > _MAX_LANGUAGE_CODES:
        raise ValueError(
            f"language_codes accepts at most {_MAX_LANGUAGE_CODES} codes "
            f"(got {len(normalized)} after normalization)"
        )
    if "multi" in normalized and len(normalized) > 1:
        raise ValueError(
            "'multi' routes to the unsteered multilingual model "
            "and cannot be combined with other language codes"
        )
    return normalized


# Server-side cap on `agent_context` length (mirrors the AssemblyAI streaming
# API's MAX_PROMPT_CHARS). Oversize values sent mid-stream make the server
# cancel the whole session, so the cap is enforced client-side.
_MAX_AGENT_CONTEXT_CHARS = 1750


def _validate_agent_context(agent_context: str) -> None:
    """Reject explicit agent_context longer than the server cap."""
    if len(agent_context) > _MAX_AGENT_CONTEXT_CHARS:
        raise ValueError(
            f"agent_context exceeds maximum length of {_MAX_AGENT_CONTEXT_CHARS} "
            f"characters (got {len(agent_context)})"
        )


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 16000,
        encoding: Literal["pcm_s16le", "pcm_mulaw"] = "pcm_s16le",
        model: Literal[
            "universal-streaming-english",
            "universal-streaming-multilingual",
            "u3-rt-pro",
            "u3-rt-pro-beta-1",
            "universal-3-5-pro",
            "universal-3-6-pro",
        ] = "universal-3-6-pro",
        language_detection: NotGivenOr[bool] = NOT_GIVEN,
        language_codes: NotGivenOr[str | list[str]] = NOT_GIVEN,
        end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_turn_silence: NotGivenOr[int] = NOT_GIVEN,
        max_turn_silence: NotGivenOr[int] = NOT_GIVEN,
        format_turns: NotGivenOr[bool] = NOT_GIVEN,
        continuous_partials: NotGivenOr[bool] = NOT_GIVEN,
        interruption_delay: NotGivenOr[int] = NOT_GIVEN,
        keyterms_prompt: NotGivenOr[list[str]] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        agent_context: NotGivenOr[str] = NOT_GIVEN,
        previous_context_n_turns: NotGivenOr[int] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        speaker_labels: NotGivenOr[bool] = NOT_GIVEN,
        max_speakers: NotGivenOr[int] = NOT_GIVEN,
        domain: NotGivenOr[str] = NOT_GIVEN,
        voice_focus: NotGivenOr[Literal["near-field", "far-field"]] = NOT_GIVEN,
        voice_focus_threshold: NotGivenOr[float] = NOT_GIVEN,
        mode: NotGivenOr[Literal["min_latency", "balanced", "max_accuracy"]] = NOT_GIVEN,
        buffer_size_seconds: float = 0.05,
        base_url: str = "wss://streaming.assemblyai.com",
    ):
        """
        Args:
            base_url: The AssemblyAI streaming endpoint base URL. Use the EU endpoint
                (wss://streaming.eu.assemblyai.com) for streaming in the EU. Defaults to
                wss://streaming.assemblyai.com.
                See https://www.assemblyai.com/docs/universal-streaming for more details.
            vad_threshold: The threshold for voice activity detection (VAD). A value between
                0 and 1 that determines how sensitive the VAD is. Lower values make the VAD
                more sensitive (detects quieter speech). Higher values make it less sensitive.
                Defaults to 0.4.
            language_codes: Steer transcription toward one or more expected languages.
                Accepts a single code ('es') or a list (['en', 'es']). Each entry
                accepts any common format ('en', 'en-US', 'english') and is normalized
                to a bare ISO 639-1 code before being sent; duplicates after
                normalization are dropped, preserving order. One code biases the model
                toward that language — several codes, toward that set — instead of
                automatically detecting/code-switching across all supported languages.
                At most 10 codes; 'multi' (the unsteered multilingual default) cannot
                be combined with other codes. Leave unset to use the model's default
                multilingual behavior. Only supported with the Universal-3 Pro family
                models. Can be updated mid-session via ``update_options``; pass an
                empty list (or empty string) there to clear steering back to the model
                default. At construction an empty value is equivalent to leaving it
                unset.
            min_turn_silence: Minimum silence in ms before a confident end-of-turn is finalized.
            continuous_partials: Whether to emit additional partial transcripts during long
                turns at a steady ~3 second cadence, on top of the baseline partials
                (one at 750 ms after turn start, configurable via `interruption_delay`,
                and one each time silence exceeds `min_turn_silence` without ending the
                turn). Leave unset to use AssemblyAI's server defaults: enabled, except
                when `speaker_labels` is on, where the server disables it so turns break
                cleanly at speaker changes. Only supported with the Universal-3 Pro
                family models.
            interruption_delay: How soon the first early partial is emitted, in ms.
                Range 0–1000, default 500. Lower values produce faster time-to-first-token
                for barge-in; higher values produce more confident first partials. Only
                supported with the Universal-3 Pro family models.
            agent_context: Free-text context describing what the agent said, used to bias
                transcription of the user's reply. Set at construction or updated per-turn
                via `update_options(agent_context=...)`. Only supported with the
                Universal-3 Pro family models (max 1750 characters; longer values raise
                ValueError). When chat-context carryover is on (the default on the Universal-3
                Pro family) each assistant reply replaces this value automatically; turn it off
                with ``VoiceSession(stt_context_options={"forward_chat_context": False})`` to
                manage this manually.
            previous_context_n_turns: Maximum number of prior conversation entries (user
                transcripts and any `agent_context` values) carried forward as context for
                each transcription. Set to 0 to disable automatic context carryover
                entirely; leave unset to use the server default (recommended). Range 0–100.
                Only supported with the Universal-3 Pro family models. Set at construction
                (connect) time only; it cannot be changed via `update_options`.
            voice_focus: Voice Focus isolates the primary voice and suppresses background
                noise (chatter, keyboard clicks, fan hum, room echo) before the audio reaches
                the model. Use 'near-field' for headsets, handsets, and close-talking
                microphones; use 'far-field' for conference rooms, laptop mics, and other
                distant-mic setups. Only supported with the Universal-3 Pro family models.
                Set at construction (connect) time only.
                See https://www.assemblyai.com/docs/streaming/voice-focus.
            voice_focus_threshold: Controls how aggressively background audio is suppressed,
                a float between 0.0 and 1.0 (higher is more aggressive). Only takes effect
                alongside `voice_focus`. Only supported with the Universal-3 Pro family
                models. Set at construction (connect) time only.
            mode: Accuracy/latency preset for the Universal-3 Pro family: 'min_latency'
                (fastest time-to-text), 'balanced' (the server default, recommended for
                voice agents), or 'max_accuracy' (highest accuracy, for scribes/post-call).
                The model applies its own per-mode silence tuning. To let that tuning take
                effect, the plugin suppresses its default 100ms min/max turn-silence windows
                when a mode is set; values you pass explicitly for `min_turn_silence` /
                `max_turn_silence` still take precedence over the mode's defaults.
                Leave unset to use the server default. Only supported with the Universal-3 Pro
                family models. Set at construction (connect) time only.
        """
        # An explicit empty value ([] or "") is equivalent to unset at
        # construction — the param is omitted from the connect query either
        # way — so it is not subject to the U3-Pro-family gate below.
        if is_given(language_codes) and not language_codes:
            language_codes = NOT_GIVEN
        if is_given(agent_context):
            _validate_agent_context(agent_context)

        # chat-context carryover into agent_context is only available on the u3-rt-pro family,
        # where it is on; the session's stt_context_options.forward_chat_context toggle controls it.
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="word",
                offline_recognize=False,
                diarization=is_given(speaker_labels) and speaker_labels is True,
                keyterms=True,
                chat_context=model in _U3_PRO_MODELS,
            ),
        )

        # These parameters are only supported by the Universal-3 Pro family of models.
        if model not in _U3_PRO_MODELS:
            _u3_pro_only_params = {
                "prompt": prompt,
                "agent_context": agent_context,
                "previous_context_n_turns": previous_context_n_turns,
                "continuous_partials": continuous_partials,
                "interruption_delay": interruption_delay,
                "voice_focus": voice_focus,
                "voice_focus_threshold": voice_focus_threshold,
                "mode": mode,
                "language_codes": language_codes,
            }
            for _param_name, _param_value in _u3_pro_only_params.items():
                if is_given(_param_value):
                    raise ValueError(
                        f"The {_param_name!r} parameter is only supported with the "
                        f"{', '.join(_U3_PRO_MODELS)} models."
                    )

        self._base_url = base_url
        assemblyai_api_key = api_key if is_given(api_key) else os.environ.get("ASSEMBLYAI_API_KEY")
        if not assemblyai_api_key:
            raise ValueError(
                "AssemblyAI API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `ASSEMBLYAI_API_KEY` environment variable"
            )
        self._api_key = assemblyai_api_key

        # we want to minimize latency as much as possible, it's ok if the phrase arrives in multiple final transcripts
        # designed to work with LK's end of turn models.
        # Skip this default when a `mode` preset is selected so the server's
        # per-mode silence tuning governs instead of being overridden by 100.
        if not is_given(min_turn_silence) and not is_given(mode):
            min_turn_silence = 100

        # Normalize to bare ISO 639-1 codes (e.g. "es-ES" / "Spanish" -> "es"),
        # the form AssemblyAI's language steering expects.
        normalized_language_codes: NotGivenOr[list[str]] = NOT_GIVEN
        if is_given(language_codes):
            normalized_language_codes = _normalize_language_codes(language_codes)

        self._opts = STTOptions(
            sample_rate=sample_rate,
            buffer_size_seconds=buffer_size_seconds,
            encoding=encoding,
            speech_model=model,
            language_detection=language_detection,
            language_codes=normalized_language_codes,
            end_of_turn_confidence_threshold=end_of_turn_confidence_threshold,
            min_turn_silence=min_turn_silence,
            max_turn_silence=max_turn_silence,
            format_turns=format_turns,
            continuous_partials=continuous_partials,
            interruption_delay=interruption_delay,
            keyterms_prompt=keyterms_prompt,
            prompt=prompt,
            agent_context=agent_context,
            previous_context_n_turns=previous_context_n_turns,
            vad_threshold=vad_threshold,
            speaker_labels=speaker_labels,
            max_speakers=max_speakers,
            domain=domain,
            voice_focus=voice_focus,
            voice_focus_threshold=voice_focus_threshold,
            mode=mode,
        )
        # user keyterms; _opts.keyterms_prompt holds the effective set (user + session)
        self._user_keyterms: list[str] = list(keyterms_prompt or [])
        self._session_keyterms: list[str] = []
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.speech_model

    @property
    def provider(self) -> str:
        return "AssemblyAI"

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Not implemented")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = dataclasses.replace(self._opts)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            base_url=self._base_url,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
        end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_turn_silence: NotGivenOr[int] = NOT_GIVEN,
        max_turn_silence: NotGivenOr[int] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        agent_context: NotGivenOr[str] = NOT_GIVEN,
        keyterms_prompt: NotGivenOr[list[str]] = NOT_GIVEN,
        language_codes: NotGivenOr[str | list[str]] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        continuous_partials: NotGivenOr[bool] = NOT_GIVEN,
        interruption_delay: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        # Validate/normalize before mutating any option so a ValueError from a
        # bad value cannot leave _opts partially updated.
        if is_given(language_codes):
            if self._opts.speech_model not in _U3_PRO_MODELS:
                raise ValueError(
                    "The 'language_codes' parameter is only supported with the "
                    f"{', '.join(_U3_PRO_MODELS)} models."
                )
            language_codes = _normalize_language_codes(language_codes)
        if is_given(agent_context):
            _validate_agent_context(agent_context)

        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds
        if is_given(end_of_turn_confidence_threshold):
            self._opts.end_of_turn_confidence_threshold = end_of_turn_confidence_threshold
        if is_given(min_turn_silence):
            self._opts.min_turn_silence = min_turn_silence
        if is_given(max_turn_silence):
            self._opts.max_turn_silence = max_turn_silence
        if is_given(prompt):
            self._opts.prompt = prompt
        if is_given(agent_context):
            self._opts.agent_context = agent_context
        if is_given(keyterms_prompt):
            self._user_keyterms = list(keyterms_prompt)
            # re-merge with the active session keyterms so a user update doesn't drop them
            keyterms_prompt = list(dict.fromkeys([*self._user_keyterms, *self._session_keyterms]))
            self._opts.keyterms_prompt = keyterms_prompt
        if is_given(language_codes):
            self._opts.language_codes = language_codes
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(continuous_partials):
            self._opts.continuous_partials = continuous_partials
        if is_given(interruption_delay):
            self._opts.interruption_delay = interruption_delay

        for stream in self._streams:
            stream.update_options(
                buffer_size_seconds=buffer_size_seconds,
                end_of_turn_confidence_threshold=end_of_turn_confidence_threshold,
                min_turn_silence=min_turn_silence,
                max_turn_silence=max_turn_silence,
                prompt=prompt,
                agent_context=agent_context,
                keyterms_prompt=keyterms_prompt,
                language_codes=language_codes,
                vad_threshold=vad_threshold,
                continuous_partials=continuous_partials,
                interruption_delay=interruption_delay,
            )

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        if keyterms == self._session_keyterms:
            return
        self._session_keyterms = list(keyterms)
        merged = list(dict.fromkeys([*self._user_keyterms, *keyterms]))
        self._opts.keyterms_prompt = merged
        # applied live via the stream's UpdateConfiguration (no reconnect)
        for stream in self._streams:
            stream.update_options(keyterms_prompt=merged)

    def _push_conversation_item(self, role: str, text: str) -> None:
        if role == "assistant" and text:
            if len(text) > _MAX_AGENT_CONTEXT_CHARS:
                # Keep the tail: the end of the reply (the question posed to
                # the user) has the most biasing value for their next utterance.
                logger.debug(
                    "truncating agent_context carryover from %d to %d chars",
                    len(text),
                    _MAX_AGENT_CONTEXT_CHARS,
                )
                text = text[-_MAX_AGENT_CONTEXT_CHARS:]
            self.update_options(agent_context=text)


class SpeechStream(
    stt.WebSocketStream[_Begin | _SpeechStarted | _Turn | _Termination | _Error, _Out]
):
    completion: ClassVar[stt.Completion] = "session_finished"
    # SpeechStarted and end_of_turn are AssemblyAI's own speech boundaries
    provider_vad = True

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        base_url: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._base_url = base_url
        self.frame_ms = round(opts.buffer_size_seconds * 1000)
        self._last_preflight_start_time: float = 0
        self._session_id: str | None = None
        self._expires_at: int | None = None
        self._anchored = False

    @property
    def session_id(self) -> str | None:
        """The AssemblyAI session ID. Set when the WebSocket connection is established
        (before any speech events). None until the connection completes.
        Share this with the AssemblyAI team when reporting issues."""
        return self._session_id

    @property
    def expires_at(self) -> int | None:
        """Unix timestamp when the AssemblyAI session expires. Set alongside session_id
        when the WebSocket connection is established."""
        return self._expires_at

    def update_options(
        self,
        *,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
        end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_turn_silence: NotGivenOr[int] = NOT_GIVEN,
        max_turn_silence: NotGivenOr[int] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        agent_context: NotGivenOr[str] = NOT_GIVEN,
        keyterms_prompt: NotGivenOr[list[str]] = NOT_GIVEN,
        language_codes: NotGivenOr[str | list[str]] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        continuous_partials: NotGivenOr[bool] = NOT_GIVEN,
        interruption_delay: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        # Validate/normalize before mutating any option so a ValueError from a
        # bad value cannot leave _opts partially updated.
        if is_given(language_codes):
            if self._opts.speech_model not in _U3_PRO_MODELS:
                raise ValueError(
                    "The 'language_codes' parameter is only supported with the "
                    f"{', '.join(_U3_PRO_MODELS)} models."
                )
            language_codes = _normalize_language_codes(language_codes)
        if is_given(agent_context):
            _validate_agent_context(agent_context)

        # the audio already chunked keeps its size
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds
        if is_given(end_of_turn_confidence_threshold):
            self._opts.end_of_turn_confidence_threshold = end_of_turn_confidence_threshold
        if is_given(min_turn_silence):
            self._opts.min_turn_silence = min_turn_silence
        if is_given(max_turn_silence):
            self._opts.max_turn_silence = max_turn_silence
        if is_given(prompt):
            self._opts.prompt = prompt
        if is_given(agent_context):
            self._opts.agent_context = agent_context
        if is_given(keyterms_prompt):
            self._opts.keyterms_prompt = keyterms_prompt
        if is_given(language_codes):
            self._opts.language_codes = language_codes
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(continuous_partials):
            self._opts.continuous_partials = continuous_partials
        if is_given(interruption_delay):
            self._opts.interruption_delay = interruption_delay

        # applied live on the open connection; the next one connects with them
        update = _Out(type="UpdateConfiguration")
        if is_given(prompt):
            update["prompt"] = prompt
        if is_given(agent_context):
            update["agent_context"] = agent_context
        if is_given(keyterms_prompt):
            update["keyterms_prompt"] = keyterms_prompt
        if is_given(language_codes):
            update["language_codes"] = language_codes
        if is_given(max_turn_silence):
            update["max_turn_silence"] = max_turn_silence
        if is_given(min_turn_silence):
            update["min_turn_silence"] = min_turn_silence
        if is_given(end_of_turn_confidence_threshold):
            update["end_of_turn_confidence_threshold"] = end_of_turn_confidence_threshold
        if is_given(continuous_partials):
            update["continuous_partials"] = continuous_partials
        if is_given(interruption_delay):
            update["interruption_delay"] = interruption_delay
        if is_given(vad_threshold):
            update["vad_threshold"] = vad_threshold
        if len(update) > 1:
            self.send_now(update)

    def force_endpoint(self) -> None:
        """Force-finalize the current turn immediately."""
        self.send_now(_Out(type="ForceEndpoint"))

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        # Universal-3 Pro family defaults: min=100, max=min (so both 100 unless overridden).
        # When a `mode` preset is selected, leave them unset (None) unless the
        # caller set them explicitly, so the server's per-mode silence tuning is
        # not overridden by the latency-optimized 100ms default.
        min_silence: int | None
        max_silence: int | None
        if self._opts.speech_model in _U3_PRO_MODELS:
            default_min = None if is_given(self._opts.mode) else 100
            min_silence = (
                self._opts.min_turn_silence
                if is_given(self._opts.min_turn_silence)
                else default_min
            )
            max_silence = (
                self._opts.max_turn_silence
                if is_given(self._opts.max_turn_silence)
                else min_silence
            )
        else:
            min_silence = (
                self._opts.min_turn_silence if is_given(self._opts.min_turn_silence) else None
            )
            max_silence = (
                self._opts.max_turn_silence if is_given(self._opts.max_turn_silence) else None
            )

        live_config = {
            "sample_rate": self._opts.sample_rate,
            "encoding": self._opts.encoding,
            "speech_model": self._opts.speech_model,
            "format_turns": self._opts.format_turns if is_given(self._opts.format_turns) else None,
            "continuous_partials": self._opts.continuous_partials
            if is_given(self._opts.continuous_partials)
            else None,
            "interruption_delay": self._opts.interruption_delay
            if is_given(self._opts.interruption_delay)
            else None,
            "end_of_turn_confidence_threshold": self._opts.end_of_turn_confidence_threshold
            if is_given(self._opts.end_of_turn_confidence_threshold)
            else None,
            "min_turn_silence": min_silence,
            "max_turn_silence": max_silence,
            "keyterms_prompt": json.dumps(self._opts.keyterms_prompt)
            if self._opts.keyterms_prompt
            else None,
            "language_detection": self._opts.language_detection
            if is_given(self._opts.language_detection)
            else True
            if "multilingual" in self._opts.speech_model
            or self._opts.speech_model in _U3_PRO_MODELS
            else False,
            "language_codes": json.dumps(self._opts.language_codes)
            if is_given(self._opts.language_codes) and self._opts.language_codes
            else None,
            "prompt": self._opts.prompt if is_given(self._opts.prompt) else None,
            "agent_context": self._opts.agent_context
            if is_given(self._opts.agent_context)
            else None,
            "previous_context_n_turns": self._opts.previous_context_n_turns
            if is_given(self._opts.previous_context_n_turns)
            else None,
            "vad_threshold": self._opts.vad_threshold
            if is_given(self._opts.vad_threshold)
            else None,
            "speaker_labels": self._opts.speaker_labels
            if is_given(self._opts.speaker_labels)
            else None,
            "max_speakers": self._opts.max_speakers if is_given(self._opts.max_speakers) else None,
            "domain": self._opts.domain if is_given(self._opts.domain) else None,
            "voice_focus": self._opts.voice_focus if is_given(self._opts.voice_focus) else None,
            "voice_focus_threshold": self._opts.voice_focus_threshold
            if is_given(self._opts.voice_focus_threshold)
            else None,
            "mode": self._opts.mode if is_given(self._opts.mode) else None,
        }
        filtered_config = {
            k: ("true" if v else "false") if isinstance(v, bool) else v
            for k, v in live_config.items()
            if v is not None
        }
        # SpeechStarted's timestamp counts from this connection's first audio
        self._anchored = False
        base_url = self._base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return stt.WebSocketTarget(
            f"{base_url}/v3/ws?{urlencode(filtered_config)}",
            headers={
                "Authorization": self._api_key,
                "Content-Type": "application/json",
                "User-Agent": "AssemblyAI/1.0 (integration=Livekit)",
            },
        )

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | _Out | None:
        match item:
            case rtc.AudioFrame():
                if not self._anchored:
                    self.start_time = time.time()
                    self._anchored = True
                return bytes(item.data)
            case stt.Flush():
                return None
            case stt.EndOfAudio():
                return _Out(type="Terminate")
            case _:
                assert_never(item)

    def decode(
        self, frame: bytes | str
    ) -> _Begin | _SpeechStarted | _Turn | _Termination | _Error | None:
        # an untagged message is the error only when it carries one: another (one AssemblyAI
        # added since) is skipped, named once by its keys
        if isinstance(frame, str):
            try:
                fields = _OBJECT.validate_json(frame)
            except ValidationError as e:
                raise stt.protocol_error(self, f"{e}; frame: {frame}") from e
            if "type" not in fields and "error" not in fields:
                stt.log_skipped("AssemblyAI", f"untagged: {','.join(sorted(fields))}")
                return None
        return super().decode(frame)

    def on_message(self, message: _Begin | _SpeechStarted | _Turn | _Termination | _Error) -> None:
        if "error" in message:
            # a refused request: the close that follows (3005) carries the same reason
            raise APIError(
                "AssemblyAI refused the request", kind="invalid_request", detail=message["error"]
            )
        elif message["type"] == "Begin":
            self._session_id = message["id"]
            self._expires_at = message["expires_at"]
            logger.info(
                "AssemblyAI session started id=%s expires_at=%s",
                self._session_id,
                self._expires_at,
            )
        elif message["type"] == "SpeechStarted":
            # SpeechStarted can arrive well after the speech onset: its stream-relative ms are
            # the server VAD's onset, from the connection's first audio
            timestamp = message.get("timestamp")
            self.report(
                stt.SpeechStarted(
                    speech_start_time=self.start_time + timestamp / 1000
                    if timestamp is not None
                    else None
                )
            )
        elif message["type"] == "Termination":
            self.report(stt.SessionFinished())
        elif message["type"] == "Turn":
            self._turn(message)
        else:
            self._skip_unknown_message(message)

    def _turn(self, message: _Turn) -> None:
        language = LanguageCode(message.get("language_code", "en"))
        # "A", "B", ... or "UNKNOWN"
        speaker_label = message.get("speaker_label")
        speaker_id = speaker_label if speaker_label and speaker_label != "UNKNOWN" else None

        # the server's per-turn scores ride the transcripts' metadata. end_of_turn_confidence
        # rises from 0 toward 1 across the partials of a turn held open between
        # min_turn_silence and max_turn_silence (1.0 on the final); language_confidence scores
        # the language_code when language detection is on
        turn_metadata: dict[str, JsonValue] = {}
        if (end_of_turn_confidence := message.get("end_of_turn_confidence")) is not None:
            turn_metadata["end_of_turn_confidence"] = end_of_turn_confidence
        if (language_confidence := message.get("language_confidence")) is not None:
            turn_metadata["language_confidence"] = language_confidence
        metadata = turn_metadata or None

        # word timestamps are in milliseconds
        # https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api#receive.receiveTurn.words
        offset = self.start_time_offset
        words = [
            TranscriptChunk(
                text=word["text"],
                start_time=word["start"] / 1000 + offset,
                end_time=word["end"] / 1000 + offset,
                start_time_offset=offset,
                confidence=word["confidence"],
            )
            for word in message["words"]
        ]
        start_time = words[0].start_time or 0.0 if words else 0.0
        end_time = words[-1].end_time or 0.0 if words else 0.0
        confidence = sum(w.confidence or 0.0 for w in words) / len(words) if words else 0.0

        # words (interim) and transcript (final) are cumulative; utterance (preflight) is a chunk
        if words:
            self.report(
                stt.Transcript(
                    " ".join(w.text for w in words),
                    language,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    speaker_id=speaker_id,
                    words=words,
                    metadata=metadata,
                )
            )

        if utterance := message.get("utterance"):
            if self._last_preflight_start_time == 0.0:
                self._last_preflight_start_time = start_time
            chunk = [
                w
                for w in words
                if w.start_time is not None and w.start_time >= self._last_preflight_start_time
            ]
            self.report(
                stt.Transcript(
                    utterance,
                    language,
                    preflight=True,
                    start_time=self._last_preflight_start_time,
                    end_time=end_time,
                    confidence=sum(w.confidence or 0.0 for w in chunk) / max(len(chunk), 1),
                    speaker_id=speaker_id,
                    words=chunk,
                    metadata=metadata,
                )
            )
            self._last_preflight_start_time = end_time

        formatted = message.get("turn_is_formatted", False)
        if message.get("end_of_turn", False) and (
            not (is_given(self._opts.format_turns) and self._opts.format_turns) or formatted
        ):
            self.report(
                stt.Transcript(
                    message.get("transcript", ""),
                    language,
                    final=True,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    speaker_id=speaker_id,
                    words=words,
                    metadata=metadata,
                )
            )
            self.report(stt.SpeechEnded())
            self._last_preflight_start_time = 0.0
