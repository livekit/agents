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
    """Inference Fallback Adapter: configuration for a fallback STT model that runs server-side in LiveKit Inference, providing automatic fallback between providers.

    Extra fields are passed through to the provider.

    Example:
        >>> FallbackModel(model="deepgram/nova-3", extra_kwargs={"keyterm": ["livekit"]})
    """

    model: str
    """Model name (e.g. "deepgram/nova-3", "assemblyai/universal-streaming", "cartesia/ink-whisper")."""
    extra_kwargs: NotRequired[dict[str, object]]
    """Extra configuration for the model."""


DeepgramModels: TypeAlias = Literal["deepgram/nova-3", "deepgram/nova-3-medical", "deepgram/nova-2", "deepgram/nova-2-medical", "deepgram/nova-2-conversationalai", "deepgram/nova-2-phonecall"]


DeepgramFluxModels: TypeAlias = Literal["deepgram/flux-general", "deepgram/flux-general-en", "deepgram/flux-general-multi"]


CartesiaModels: TypeAlias = Literal["cartesia/ink-whisper", "cartesia/ink-2"]


AssemblyAIModels: TypeAlias = Literal["assemblyai/universal-streaming", "assemblyai/universal-streaming-multilingual", "assemblyai/u3-rt-pro", "assemblyai/universal-3-5-pro", "assemblyai/universal-3-6-pro"]


XaiModels: TypeAlias = Literal["xai/stt-1"]


SpeechmaticsModels: TypeAlias = Literal["speechmatics/enhanced", "speechmatics/standard", "speechmatics/linden-1"]


InworldModels: TypeAlias = Literal["inworld/inworld-stt-1"]


GoogleModels: TypeAlias = Literal["google/gemini-3.5-transcribe-live"]


AutoModels: TypeAlias = Literal["auto"]
"""The gateway picks the model."""


STTLanguages: TypeAlias = Literal["multi", "en", "de", "es", "fr", "ja", "pt", "zh", "hi"]


STTEncoding: TypeAlias = Literal["pcm_s16le"]


STTModels: TypeAlias = (
    DeepgramModels
    | DeepgramFluxModels
    | CartesiaModels
    | AssemblyAIModels
    | XaiModels
    | SpeechmaticsModels
    | InworldModels
    | GoogleModels
    | AutoModels
)
