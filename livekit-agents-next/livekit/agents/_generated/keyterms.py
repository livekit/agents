# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/turn/keyterms.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from ..llm import LLM


class KeytermDetectionOptions(TypedDict, total=False):
    """Configuration for automatic keyterm detection.

    Lives under the ``keyterm_detection`` key of :class:`STTContextOptions`. Absent or
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


@dataclass(frozen=True, kw_only=True, slots=True)
class ResolvedKeytermDetection:
    """The resolved detection config as the host receives it: `model` is `None` for a caller-supplied
    `LLM`, and `instructions` is always the resolved string.
    """

    enabled: bool
    turn_interval: int
    max_keyterms: int | None
    timeout: float
    instructions: str
    model: str | None
    """The model to build, or `None` for "the caller supplied an LLM; use it"."""
    tool_name: str


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


DetectionLlm: TypeAlias = str | object
"""`LLM | str`; an instance stays host-side and crosses as "use the one given", distinct from
`None`, "none configured".
"""


KeytermDetectionOptions.__module__ = "livekit.agents.voice.agent_session"
ResolvedKeytermDetection.__module__ = "livekit.agents.voice.agent_session"
STTContextOptions.__module__ = "livekit.agents.voice.agent_session"
