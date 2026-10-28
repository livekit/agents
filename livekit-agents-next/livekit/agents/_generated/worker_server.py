# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/worker.rs
#     rust/crates/lk-agents-abi/src/worker/server_host.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, kw_only=True, slots=True)
class RegisteredInfo:
    """Registration accepted by the server, delivered as a typed record."""

    worker_id: str
    """The id the server issued."""
    region: str
    protocol: int
    """The protocol version the server chose."""
    deployment: str
    agent_name: str


@dataclass(frozen=True, kw_only=True, slots=True)
class JobAnswerAccept:
    name: str
    identity: str
    metadata: str
    attributes: list[tuple[str, str]]


@dataclass(frozen=True, kw_only=True, slots=True)
class JobAnswerReject:
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class JobAnswerTerminate:
    """Reject and ask the server to end the job."""


@dataclass(frozen=True, kw_only=True, slots=True)
class JobAnswerUnanswered:
    """The callback returned without answering: reject without terminating."""


JobAnswer: TypeAlias = JobAnswerAccept | JobAnswerReject | JobAnswerTerminate | JobAnswerUnanswered
"""The host's answer to a job request."""


@dataclass(frozen=True, kw_only=True, slots=True)
class EndpointRoute:
    """One route of the endpoint manifest the worker registers."""

    path: str
    methods: list[str]
    kind: str
    """`"http"`, or `"text"` for a session endpoint."""
    public: bool
