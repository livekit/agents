# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/inference/fallback.rs

from __future__ import annotations

from typing import NotRequired, TypedDict

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
