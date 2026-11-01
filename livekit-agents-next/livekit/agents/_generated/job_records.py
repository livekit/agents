# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/job.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum, unique
from inspect import get_annotations

from livekit.protocol.agent import Job


@dataclass
class JobAcceptArguments:
    name: str
    identity: str
    metadata: str
    attributes: dict[str, str] | None = None


@dataclass
class RunningJobInfo:
    accept_arguments: JobAcceptArguments
    job: Job
    url: str
    token: str
    worker_id: str
    fake_job: bool
    received_at: float = 0.0
    """The worker received the availability request."""
    accepted_at: float = 0.0
    """The request handler accepted the job."""
    assigned_at: float = 0.0
    """The server's assignment (room token) arrived."""
    launched_at: float = 0.0
    """A process was acquired from the pool and handed the job."""


@unique
class JobExecutorType(Enum):
    PROCESS = "process"
    THREAD = "thread"


@unique
class AutoSubscribe(StrEnum):
    SUBSCRIBE_ALL = "subscribe_all"
    SUBSCRIBE_NONE = "subscribe_none"
    AUDIO_ONLY = "audio_only"
    VIDEO_ONLY = "video_only"


JobAcceptArguments.__annotations__ = get_annotations(JobAcceptArguments, eval_str=True)
JobAcceptArguments.__module__ = "livekit.agents.job"
RunningJobInfo.__annotations__ = get_annotations(RunningJobInfo, eval_str=True)
RunningJobInfo.__module__ = "livekit.agents.job"
JobExecutorType.__module__ = "livekit.agents.job"
AutoSubscribe.__module__ = "livekit.agents.job"
