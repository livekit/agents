"""A job runner as the worker builds one: the job process launcher behind its supervisor."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from livekit.agents import JobContext, _core
from livekit.agents.ipc import JobLauncher, JobRunnerOptions
from livekit.agents.job import JobAcceptArguments, RunningJobInfo
from livekit.protocol import agent

Entrypoint = Callable[[JobContext], Awaitable[None]]


def job_runner(
    entrypoint: Entrypoint,
    *,
    session_end: Entrypoint | None = None,
    env: dict[str, str] | None = None,
    thread: bool = True,
) -> _core.JobSupervisor:
    """A runner that hosts one job at a time in a thread (or a process) of its own; ``env`` is
    the server's LiveKit settings the job uses over the process environment."""
    launcher = JobLauncher(
        job_entrypoint_fnc=entrypoint,
        session_end_fnc=session_end,
        simulation_end_fnc=None,
        session_end_timeout=5,
        env=tuple((env or {}).items()),
        loop=asyncio.get_running_loop(),
    )
    options = JobRunnerOptions(
        thread=thread,
        initialize_timeout=10,
        close_timeout=10,
        ping_interval=0,
        high_ping_threshold=1,
    )
    return _core.JobSupervisor(launcher, options)


def running_job(job: agent.Job, *, fake_job: bool = True) -> RunningJobInfo:
    """The worker's record of ``job``, as a dispatch hands it over."""
    return RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="", identity="agent", metadata=""),
        job=job,
        url="",
        token="",
        worker_id="",
        fake_job=fake_job,
        received_at=None,
        accepted_at=None,
        assigned_at=None,
    )
