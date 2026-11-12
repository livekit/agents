"""Runs one thread job while the `livekit.agents` log handler is stalled, then checks the job's
exit line once the handler resumes. Run as `python -m tests_new.host.stalled_log_job` in a fresh
interpreter, so the job runner is the first to start the process-wide log forwarding."""

from __future__ import annotations

import asyncio
import logging
import threading

from livekit.agents import JobContext
from livekit.protocol import agent, models

from .ipc import job_runner, running_job


class Stalling(logging.Handler):
    """Holds the core's "job thread initialized" line until resumed, outside the handler's
    lock; keeps every record."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.stalled = threading.Event()
        self.resume = threading.Event()
        self.records: list[logging.LogRecord] = []

    def handle(self, record: logging.LogRecord) -> bool:
        if record.getMessage() == "job thread initialized":
            self.stalled.set()
            if not self.resume.wait(timeout=10):
                raise TimeoutError("the test never released the log handler")
        return super().handle(record)

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


async def main() -> None:
    handler = Stalling()
    logger = logging.getLogger("livekit.agents")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    ran = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def entrypoint(ctx: JobContext) -> None:
        loop.call_soon_threadsafe(ran.set)
        ctx.shutdown("done ✓")

    runner = job_runner(entrypoint)
    await runner.start()
    await runner.initialize()
    job = agent.Job(id="AJ_logging", room=models.Room(name="logging"))
    await runner.launch_job(running_job(job))
    async with asyncio.timeout(5):
        await ran.wait()  # the job ran while the log handler was stalled
        await runner.join()
    assert handler.stalled.is_set(), "nothing was logged"
    handler.resume.set()

    async with asyncio.timeout(5):
        while not any(r.getMessage() == "job thread exiting" for r in handler.records):
            await asyncio.sleep(0.01)
    exiting = next(r for r in handler.records if r.getMessage() == "job thread exiting")
    fields = {key: getattr(exiting, key, None) for key in ("reason", "job_id", "room")}
    assert fields == {"reason": "done ✓", "job_id": "AJ_logging", "room": "logging"}, fields
    await runner.aclose()


if __name__ == "__main__":
    asyncio.run(main())
