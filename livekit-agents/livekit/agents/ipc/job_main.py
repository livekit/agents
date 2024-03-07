from .client import IPCClient
from typing import Callable
import asyncio

def _run_job(job_id: str, usercb: Callable) -> None:
    async def _run_job_async() -> None:
        client = IPCClient(job_id)
        await client.start()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_run_job_async())
