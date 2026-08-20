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

"""Typed schema for Inworld's ``providerData`` Realtime API extensions.

These are ``TypedDict``s (plain dicts at runtime) so they serialize verbatim into the
``session.update`` payload. All keys are optional; omit a key to inherit the server default.
See https://docs.inworld.ai/realtime/provider-data for the field-by-field reference.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class STTProviderData(TypedDict, total=False):
    """``providerData.stt`` — STT tuning. Hot-swappable."""

    prompt: str
    voice_profile: bool
    language_hints: list[str]
    end_of_turn_confidence_threshold: float
    vad_threshold: float
    min_end_of_turn_silence: int
    max_turn_silence: int


class TTSProviderData(TypedDict, total=False):
    """``providerData.tts`` — segmentation, language, delivery, alignment."""

    segmenter_strategy: Literal[
        "auto", "balanced", "sentence", "full_turn", "fast_start", "per_segment_context"
    ]
    steering_handling: Literal["repeat_each_chunk", "emit_once"]
    language: str
    delivery_mode: Literal["STABLE", "BALANCED", "CREATIVE"]
    conversational: bool  # locked at session open
    user_turn_mode: Literal["both", "audio_only", "text_only", "none"]  # locked at session open
    timestamp_type: Literal["WORD", "CHARACTER"]
    timestamp_transport_strategy: Literal["SYNC", "ASYNC"]


class MemoryProviderData(TypedDict, total=False):
    """``providerData.memory`` — automatic conversation memory and summarization."""

    enabled: bool
    turn_interval: int
    max_memory_length: int
    max_transcript_items: int
    max_facts: int
    trim_after_summarize: bool


class BackchannelProviderData(TypedDict, total=False):
    """``providerData.backchannel`` — short interjections while the user speaks."""

    enabled: bool
    small_model: str
    eval_interval_ms: int
    min_speech_ms: int
    min_gap_ms: int
    max_per_turn: int
    hard_deadline_ms: int
    history_tail_items: int
    temperature: float
    max_tokens: int
    volume_gain: float
    require_pause: bool
    allowed_phrases: list[str]
    prompt_template: str
    decider_kind: str


class ResponsivenessProviderData(TypedDict, total=False):
    """``providerData.responsiveness`` — filler audio while the LLM warms up."""

    enabled: bool
    small_model: str
    initial_wait_timeout_ms: int
    hard_deadline_ms: int
    history_tail_items: int
    temperature: float
    max_tokens: int
    min_filler_gap_ms: int
    max_initial_per_turn: int
    max_buffer_deltas: int
    enable_filler_on_first_assistant_reply: bool
    prompt_template: str
    pause_text: str


class CachingProviderData(TypedDict, total=False):
    """``providerData.caching`` — explicit prompt caching for instructions/tools."""

    enabled: bool
    ttl: str
    cache_instructions: bool
    cache_tools: bool


class ReasoningConfig(TypedDict, total=False):
    effort: Literal["NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH"]
    maxTokens: int
    exclude: bool


class TextGenerationConfig(TypedDict, total=False):
    """LLM generation parameters (camelCase on the wire, per Inworld)."""

    reasoning: ReasoningConfig
    maxNewTokens: int
    temperature: float
    topP: float
    frequencyPenalty: float
    presencePenalty: float
    repetitionPenalty: float
    stopSequences: list[str]
    seed: int
    logitBias: list[dict[str, Any]]


class ProviderData(TypedDict, total=False):
    """Root ``providerData`` object merged into the Inworld session config."""

    auto_tool_response: bool
    stt: STTProviderData
    tts: TTSProviderData
    memory: MemoryProviderData
    backchannel: BackchannelProviderData
    responsiveness: ResponsivenessProviderData
    caching: CachingProviderData
    text_generation_config: TextGenerationConfig
    user_id: str
    metadata: dict[str, str]
