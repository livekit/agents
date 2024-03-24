from __future__ import annotations

import asyncio
import logging

from livekit import rtc
from livekit.protocol import agent, worker

from ..log import process_logger
from . import protocol
from .consts import START_TIMEOUT
from .job_context import JobContext


def _run_job(cch: protocol.ProcessPipe, args: protocol.JobMainArgs) -> None:
    protocol.write_msg(cch, protocol.Pong(last_timestamp=200, timestamp=200))
    protocol.write_msg(cch, protocol.Log(level=logging.INFO, message="running job"))

    return

    async def _run_job() -> None:
        client, rx = IPCClient.create(args.job_id, loop)
        logger = logging.getLogger()
        logger.addHandler(client.log_handler)

        try:
            room = rtc.Room()
            try:
                # We don't need any retry logic for the Room connection,
                # the Rust SDK is responsible for that.
                await room.connect(args.url, args.token)
            except Exception as e:
                await client.send(
                    worker.IPCJobMessage(
                        start_job=worker.StartJobResponse(error=str(e)),
                    )
                )
                return

            try:
                async with asyncio.timeout(START_TIMEOUT):
                    msg = await rx.get()
                    if not msg.HasField("start_job"):
                        process_logger.error("expected start job message")
                        return

                    start_job = msg.start_job
            except asyncio.TimeoutError:
                process_logger.error("timeout waiting for start job message")
                return

            job = start_job.job

            participant: rtc.RemoteParticipant | None = None
            if job.participant:
                participant = room.participants.get(job.participant)
                if not participant and job.type == agent.JobType.JT_PUBLISHER:
                    # the participant is not found, so the job is not valid anymore.
                    # we should just exit
                    await client.send(
                        worker.IPCJobMessage(
                            start_job=worker.StartJobResponse(
                                error=f"participant {job.participant.identity} not found",
                            )
                        )
                    )
                    await room.disconnect()
                    return

            job_ctx = JobContext(client, job, room, participant)

            async def _main_job_task():
                try:
                    await args.target(job_ctx)
                except Exception as e:
                    if e is not asyncio.CancelledError:
                        await job_ctx.shutdown()

            job_ctx.create_task(_main_job_task())

            while True:
                msg = await rx.get()
                process_logger.debug(f"received message: {msg}")

                if msg.HasField("shutdown"):
                    break

        finally:
            await client.aclose()

    loop.run_until_complete(_run_job())
