# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/turn/keyterms.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing_extensions import TypedDict

from ..llm import LLM


class KeytermDetectionOptions(TypedDict, total=False):
    """Configuration for automatic keyterm detection.

    Lives under the ``keyterm_detection`` key of :class:`KeytermsOptions`. Absent or
    ``{"enabled": False}`` keeps detection off.
    """

    enabled: bool
    """Whether to run the background detector. Defaults to ``False``."""
    llm: LLM | str | None
    """LLM used for extraction. An ``LLM`` instance, or a model string (e.g.
    ``"google/gemini-3.5-flash"``) resolved via the inference gateway. Defaults to a
    built-in detection model; the agent's own LLM is not used.
    """
    turn_interval: int
    """Run a pass once per N user turns. Defaults to ``1``."""
    max_keyterms: int | None
    """Cap on the confirmed (applied) detected keyterms if provided. Defaults to ``None``."""
    instructions: str | None
    """Override the built-in extraction prompt."""
    timeout: float
    """Seconds a single detection pass may run before it is dropped (no keyterm change).
    Defaults to ``10.0``. Raise it if a slow detection ``llm`` needs longer.
    """


class STTContextOptions(TypedDict, total=False):
    """Conversation-aware context for the STT.

    Can be passed as a plain dict::

        AgentSession(
            stt_context_options={
                "keyterms": ["LiveKit", "Acme Corp"],
                "keyterm_detection": {"enabled": True, "turn_interval": 1},
                "forward_chat_context": True,
            },
        )
    """

    keyterms: list[str]
    """Static keyterms applied wherever the STT accepts a term list; never touched by detection."""
    keyterm_detection: KeytermDetectionOptions
    """LLM-based keyterm extraction, for STTs that accept a term list."""
    forward_chat_context: bool
    """Forward conversation turns to STTs that consume context directly (e.g. AssemblyAI u3-pro).
    Defaults to ``True``; only STTs that advertise the ``chat_context`` capability act on it.
    """


class KeytermsOptions(TypedDict, total=False):
    """Keyterm biasing for STTs that accept a term list.

    .. deprecated:: v2.0
        Use :class:`STTContextOptions` (``AgentSession(stt_context_options=...)``) instead;
        its ``keyterms`` and ``keyterm_detection`` keys are identical.
    """

    keyterms: list[str]
    """Static keyterms applied wherever the STT accepts a term list; never touched by detection."""
    keyterm_detection: KeytermDetectionOptions
    """LLM-based keyterm extraction, for STTs that accept a term list."""


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
KeytermDetectionOptions.__module__ = "livekit.agents.voice.keyterm_detection"
STTContextOptions.__module__ = "livekit.agents.voice.keyterm_detection"
KeytermsOptions.__module__ = "livekit.agents.voice.keyterm_detection"
