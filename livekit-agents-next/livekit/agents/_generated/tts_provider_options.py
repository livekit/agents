# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/inference/tts_client.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from typing import Literal


class CartesiaOptions(TypedDict, total=False):
    emotion: str
    speed: Literal['slow', 'normal', 'fast'] | float
    volume: float
    duration: float
    max_buffer_delay_ms: int
    add_timestamps: bool
    add_phoneme_timestamps: bool
    use_normalized_timestamps: bool


class DeepgramOptions(TypedDict, total=False):
    mip_opt_out: bool


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
    delivery_mode: Literal['DELIVERY_MODE_UNSPECIFIED', 'STABLE', 'BALANCED', 'CREATIVE']
    timestamp_type: Literal['TIMESTAMP_TYPE_UNSPECIFIED', 'WORD', 'CHARACTER']
    apply_text_normalization: Literal['APPLY_TEXT_NORMALIZATION_UNSPECIFIED', 'ON', 'OFF']


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
    latency: Literal['normal', 'balanced', 'low']
    chunk_length: int
    max_new_tokens: int
    min_chunk_length: int
    condition_on_previous_chunks: bool
    early_stop_threshold: float


CartesiaOptions.__module__ = "livekit.agents.inference.tts"
DeepgramOptions.__module__ = "livekit.agents.inference.tts"
RimeOptions.__module__ = "livekit.agents.inference.tts"
InworldOptions.__module__ = "livekit.agents.inference.tts"
XaiOptions.__module__ = "livekit.agents.inference.tts"
FishAudioOptions.__module__ = "livekit.agents.inference.tts"
