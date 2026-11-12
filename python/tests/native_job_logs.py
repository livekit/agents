"""Spawn a job whose structured log must reach the supervisor's stdout."""

import asyncio
import logging
import multiprocessing as mp
import sys

from livekit.agents import Agent, _core, cli
from livekit.agents.ipc import JobLauncher, JobRunnerOptions
from livekit.agents.job import JobAcceptArguments, RunningJobInfo
from livekit.protocol import agent, models


async def entrypoint(ctx):
    pii = Agent(instructions="x", name="pii")
    logging.getLogger("livekit.agents").warning("from the job", extra={"lk.pii.output": pii})
    ctx.shutdown("done")


async def main(style):
    if style != "none":
        cli.setup_logging("info")
    launcher = JobLauncher(
        job_entrypoint_fnc=entrypoint,
        session_end_fnc=None,
        simulation_end_fnc=None,
        session_end_timeout=10,
        mp_ctx=mp.get_context("spawn"),
        loop=asyncio.get_running_loop(),
    )
    proc = _core.JobSupervisor(
        launcher, JobRunnerOptions(initialize_timeout=60, close_timeout=10, high_ping_threshold=1)
    )
    await proc.start()
    await proc.initialize()
    job = agent.Job(id="AJ_logs", room=models.Room(name="logs", sid="RM_logs"))
    await proc.launch_job(
        RunningJobInfo(
            accept_arguments=JobAcceptArguments(
                name="", identity="agent", metadata="", attributes={}
            ),
            job=job,
            url="",
            token="",
            worker_id="smoke",
            fake_job=True,
            received_at=0,
            accepted_at=0,
            assigned_at=0,
        )
    )
    await proc.join()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
