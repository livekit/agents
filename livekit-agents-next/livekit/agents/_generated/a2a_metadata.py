# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/a2a/metadata.rs

from __future__ import annotations

from typing import Final

EXTENSION_URI: Final = "https://livekit.io/a2a/ext/agents/v1"
EXTENSIONS_HEADER: Final = "A2A-Extensions"
GRANT_HEADER: Final = "X-LiveKit-Thread-Grant"
"""The request header that carries a thread grant beside `Authorization`."""
AGENT_CARD_PATH: Final = "/.well-known/agent-card.json"
"""Where an endpoint publishes its agent card, under `/<endpoint>`."""
RPC_PATH: Final = "/v1"
"""Where an endpoint's JSON-RPC methods live, under `/<endpoint>`; the card's URL points here."""
CHANNEL: Final = "https://livekit.io/a2a/ext/agents/v1/channel"
KIND: Final = "https://livekit.io/a2a/ext/agents/v1/kind"
INTERRUPTED: Final = "https://livekit.io/a2a/ext/agents/v1/interrupted"
INTERRUPTION_REASON: Final = "https://livekit.io/a2a/ext/agents/v1/interruptionReason"
MERGED_INTO: Final = "https://livekit.io/a2a/ext/agents/v1/mergedInto"
REASON: Final = "https://livekit.io/a2a/ext/agents/v1/reason"
VERBATIM: Final = "https://livekit.io/a2a/ext/agents/v1/verbatim"
KIND_DELEGATION: Final = "delegation"
KIND_CHAT_CTX: Final = "chat_ctx"
ANSWER_ARTIFACT_NAME: Final = "answer"
