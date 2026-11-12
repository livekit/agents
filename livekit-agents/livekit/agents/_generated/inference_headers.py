# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/inference/mod.rs

from __future__ import annotations

from typing import Final

HEADER_SESSION_ID: Final = "X-LiveKit-Session-ID"
"""The gateway's session identification header."""
HEADER_INFERENCE_PROVIDER: Final = "X-LiveKit-Inference-Provider"
"""The provider a gateway request asks for."""
HEADER_INFERENCE_PRIORITY: Final = "X-LiveKit-Inference-Priority"
"""A request's inference class (`"low"` for a background job)."""
