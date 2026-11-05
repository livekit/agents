# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/inference/fallback.rs
#     rust/crates/lk-agents-abi/src/inference/models.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Literal, NotRequired, TypeAlias

from typing_extensions import TypedDict


class FallbackModel(TypedDict):
    """Inference Fallback Adapter: configuration for a fallback TTS model that runs server-side in LiveKit Inference, providing automatic fallback between providers.

    Extra fields are passed through to the provider.

    Example:
        >>> FallbackModel(model="cartesia/sonic", voice="")
    """

    model: str
    """Model name (e.g. "cartesia/sonic", "inworld/inworld-tts-1", "rime/coda")."""
    voice: NotRequired[str]
    """Voice to use for the model."""
    extra_kwargs: NotRequired[dict[str, object]]
    """Extra configuration for the model."""


CartesiaModels: TypeAlias = Literal["cartesia", "cartesia/sonic-3.5", "cartesia/sonic-3", "cartesia/sonic-2", "cartesia/sonic-turbo", "cartesia/sonic", "cartesia/sonic-3-latest", "cartesia/sonic-latest"]


DeepgramModels: TypeAlias = Literal["deepgram", "deepgram/aura", "deepgram/aura-2"]


RimeModels: TypeAlias = Literal["rime", "rime/coda", "rime/mistv2", "rime/mistv3", "rime/mist"]


InworldModels: TypeAlias = Literal["inworld", "inworld/inworld-tts-2", "inworld/inworld-tts-1.5-max", "inworld/inworld-tts-1.5-mini", "inworld/inworld-tts-1.5", "inworld/inworld-tts-1-max", "inworld/inworld-tts-1"]


XaiModels: TypeAlias = Literal["xai/tts-1"]


FishAudioModels: TypeAlias = Literal["fishaudio", "fishaudio/s2.1-pro", "fishaudio/s2.1-pro-free", "fishaudio/s2-pro"]


TTSEncoding: TypeAlias = Literal["pcm_s16le"]


TTSModels: TypeAlias = (
    CartesiaModels | DeepgramModels | RimeModels | InworldModels | XaiModels | FishAudioModels
)
