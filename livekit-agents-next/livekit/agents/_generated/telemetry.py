# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/telemetry/report/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from .chat_items import ChatItem
    from .usage import Usage


@dataclass(kw_only=True, slots=True)
class RecordingOptions:
    """What the session records to LiveKit Cloud. Every part is on unless the session turns it off.

    `redaction = False` never disables project-level redaction; it only declines to ask for it.
    """

    audio: bool = True
    traces: bool = True
    logs: bool = True
    transcript: bool = True
    redaction: bool = False
    """Not a recorded part: `recording_enabled` ignores it."""


@dataclass(kw_only=True, slots=True)
class EvaluationRecord:
    """One judge's judgment of the session."""

    name: str
    tag: str
    verdict: str
    reasoning: str
    instructions: str


Outcome: TypeAlias = Literal["success", "fail"]
"""The session outcome. `"fail"` bumps the outcome record's severity."""


@dataclass(kw_only=True, slots=True)
class ReportTag:
    """One tag of the session, with its optional structured metadata and when it was added."""

    name: str
    metadata: object | None
    timestamp: float
    """Unix seconds."""


@dataclass(kw_only=True, slots=True)
class ReportOutcome:
    """The session's outcome and the reason given with it."""

    outcome: Outcome
    reason: str | None


@dataclass(kw_only=True, slots=True)
class SessionReport:
    """The summary of one voice session: its transcript, usage, options, events, tags and
    evaluations, and its audio recording. One snapshot every reader shares: `on_session_end`
    (through `JobContext.make_session_report()`), the Cloud records and the recording upload.
    """

    job_id: str
    room_id: str
    room: str
    agent_name: str = ""
    """The name the agent was dispatched with."""
    options: object
    """The resolved session options, its recording options under `recording_options`."""
    events: list[object] = field(default_factory=list)
    """The session's events, as dictionaries."""
    chat_history: list[ChatItem]
    turns: dict[str, tuple[int, int, bool]] = field(default_factory=dict)
    """The span of the turn each chat item belongs to, by item id: (trace id, span id, sampled)."""
    usage: Usage
    tags: list[ReportTag] = field(default_factory=list)
    evaluations: list[EvaluationRecord] = field(default_factory=list)
    outcome: ReportOutcome | None = None
    started_at: float | None = None
    """When the session started, in unix seconds."""
    timestamp: float
    """When the report was made, in unix seconds."""
    duration: float | None = None
    """The audio recording's length, in seconds."""
    audio_recording_path: str | None = None
    audio_recording_started_at: float | None = None
    """When the audio recording started, in unix seconds."""
    sdk_version: str
