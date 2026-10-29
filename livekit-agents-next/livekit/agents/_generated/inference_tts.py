# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/inference/fallback.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import NotRequired, TypedDict

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
