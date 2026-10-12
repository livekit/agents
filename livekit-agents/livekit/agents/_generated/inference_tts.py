# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/inference/fallback.rs
#     rust/crates/lk-agents-abi/src/inference/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired


class FallbackModel(TypedDict):
    """Inference Fallback Adapter: configuration for a fallback TTS model that runs server-side in LiveKit Inference, providing automatic fallback between providers.

    Extra fields are passed through to the provider.

    Example:
        >>> FallbackModel(model="cartesia/sonic", voice="")
    """

    model: str
    """Model name (e.g. "cartesia/sonic", "elevenlabs/eleven_flash_v2", "rime/coda")."""
    voice: str
    """Voice to use for the model."""
    extra_kwargs: NotRequired[dict[str, Any]]
    """Extra configuration for the model."""


CartesiaModels = Literal[
    "cartesia",
    "cartesia/sonic-3.5",
    "cartesia/sonic-3",
    "cartesia/sonic-2",
    "cartesia/sonic-turbo",
    "cartesia/sonic",
    "cartesia/sonic-3-latest",
    "cartesia/sonic-latest",
]


DeepgramModels = Literal["deepgram", "deepgram/aura", "deepgram/aura-2"]


ElevenlabsModels = Literal[
    "elevenlabs",
    "elevenlabs/eleven_flash_v2",
    "elevenlabs/eleven_flash_v2_5",
    "elevenlabs/eleven_turbo_v2",
    "elevenlabs/eleven_turbo_v2_5",
    "elevenlabs/eleven_multilingual_v2",
    "elevenlabs/eleven_v3",
]


RimeModels = Literal["rime", "rime/coda", "rime/mistv2", "rime/mistv3", "rime/mist"]


InworldModels = Literal[
    "inworld",
    "inworld/inworld-tts-2",
    "inworld/inworld-tts-1.5-max",
    "inworld/inworld-tts-1.5-mini",
    "inworld/inworld-tts-1.5",
    "inworld/inworld-tts-1-max",
    "inworld/inworld-tts-1",
]


XaiModels = Literal["xai/tts-1"]


FishAudioModels = Literal[
    "fishaudio", "fishaudio/s2.1-pro", "fishaudio/s2.1-pro-free", "fishaudio/s2-pro"
]


TTSModels = (
    CartesiaModels
    | DeepgramModels
    | ElevenlabsModels
    | RimeModels
    | InworldModels
    | XaiModels
    | FishAudioModels
)


FallbackModelType = FallbackModel | str


class CartesiaOptions(TypedDict, total=False):
    emotion: str
    speed: Literal["slow", "normal", "fast"] | float
    volume: float
    duration: float
    max_buffer_delay_ms: int
    add_timestamps: bool
    add_phoneme_timestamps: bool
    use_normalized_timestamps: bool


class DeepgramOptions(TypedDict, total=False):
    mip_opt_out: bool


class ElevenlabsOptions(TypedDict, total=False):
    inactivity_timeout: int
    apply_text_normalization: Literal["auto", "off", "on"]
    auto_mode: bool
    enable_logging: bool
    enable_ssml_parsing: bool
    sync_alignment: bool
    language_code: str
    stability: float
    similarity_boost: float
    style: float
    speed: float
    use_speaker_boost: bool
    chunk_length_schedule: list[float]
    preferred_alignment: str


class RimeOptions(TypedDict, total=False):
    """Rime-specific WebSocket JSON query parameters.
    See: https://docs.rime.ai/api-reference/endpoint/websockets-json
    """

    speed_alpha: float
    pause_between_brackets: bool
    phonemize_between_brackets: bool
    inline_speed_alpha: str
    no_text_normalization: bool


class InworldOptions(TypedDict, total=False):
    speaking_rate: float
    temperature: float
    delivery_mode: Literal["DELIVERY_MODE_UNSPECIFIED", "STABLE", "BALANCED", "CREATIVE"]
    timestamp_type: Literal["TIMESTAMP_TYPE_UNSPECIFIED", "WORD", "CHARACTER"]
    apply_text_normalization: Literal["APPLY_TEXT_NORMALIZATION_UNSPECIFIED", "ON", "OFF"]


class XaiOptions(TypedDict, total=False):
    bit_rate: Literal[32000, 64000, 96000, 128000, 192000]
    speed: float
    optimize_streaming_latency: Literal[0, 1, 2]


class FishAudioOptions(TypedDict, total=False):
    """See https://docs.fish.audio/api-reference/endpoint/websocket/tts-live"""

    speed: float
    volume: float
    normalize_loudness: bool
    temperature: float
    top_p: float
    normalize: bool
    latency: Literal["normal", "balanced", "low"]
    chunk_length: int
    max_new_tokens: int
    min_chunk_length: int
    condition_on_previous_chunks: bool
    early_stop_threshold: float


TTSEncoding = Literal["pcm_s16le"]


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
FallbackModel.__module__ = "livekit.agents.inference.tts"
CartesiaOptions.__module__ = "livekit.agents.inference.tts"
DeepgramOptions.__module__ = "livekit.agents.inference.tts"
ElevenlabsOptions.__module__ = "livekit.agents.inference.tts"
RimeOptions.__module__ = "livekit.agents.inference.tts"
InworldOptions.__module__ = "livekit.agents.inference.tts"
XaiOptions.__module__ = "livekit.agents.inference.tts"
FishAudioOptions.__module__ = "livekit.agents.inference.tts"
