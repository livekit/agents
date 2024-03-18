from string import whitespace
from time import sleep
from livekit import rtc
from .client import IPCClient
from typing import Callable
import asyncio
import sys
import multiprocessing
import logging
import contextlib
from ..log import process_logger
from livekit.protocol import worker, agents
from ..job_context import JobContext
from .consts import START_TIMEOUT


def _run_job(job_id: str, usercb: Callable) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    send_queue = asyncio.Queue[worker.IPCJobMessage | None](32)
    recv_queue = asyncio.Queue[worker.IPCWorkerMessage](32)

    async def _run_job() -> None:
        client = IPCClient(job_id, send_queue, recv_queue, loop)
        logger = logging.getLogger()
        logger.addHandler(client.log_handler)
        client_task = asyncio.create_task(client.run())

        try:
            async with asyncio.timeout(START_TIMEOUT):
                start_job = await recv_queue.get()
                if not start_job.HasField("start_job"):
                    process_logger.error("expected start job message")
                    return

                start_job = start_job.start_job
        except asyncio.TimeoutError:
            process_logger.error("timeout waiting for start job message")
            print("timeout waiting for start job message", file=sys.stderr)
            await asyncio.sleep(1)
            exit(1)

        try:
            job = start_job.job
            room = rtc.Room()
            try:
                # We don't need any retry logic for the Room connection,
                # the Rust SDK is responsible for that.
                await room.connect(start_job.url, start_job.token)
            except Exception as e:
                await send_queue.put(
                    worker.IPCJobMessage(
                        start_job=worker.StartJobResponse(error=str(e)),
                    )
                )
                return

            participant: rtc.RemoteParticipant | None = None
            if job.participant:
                participant = room.participants.get(job.participant)
                if not participant and job.type == agents.JobType.JT_PUBLISHER:
                    # the participant is not found, so the job is not valid anymore.
                    # we should just exit
                    await send_queue.put(
                        worker.IPCJobMessage(
                            start_job=worker.StartJobResponse(
                                error=f"participant {job.participant.identity} not found",
                            )
                        )
                    )
                    await room.disconnect()
                    return

            job_ctx = JobContext(send_queue, job, room, participant)
            async def _main_job_task():
                try:
                    await usercb(job_ctx)
                except Exception as e:
                    if e is not asyncio.CancelledError:
                        await job_ctx.shutdown()

            job_ctx.create_task(_main_job_task())

            while True:
                msg = await recv_queue.get()
                process_logger.debug(f"received message: {msg}")

                if msg.HasField("shutdown"):
                    break

        finally:
            await client_task

    loop.run_until_complete(_run_job())
