"""Sends ``tasks/cancel`` for one of the booking office's tasks, with reason ``review``.

``uv run python cancel.py <thread_id> <task_id>``, with the worker's environment, while
``book_flight`` runs; both ids are in the ``backend turn`` log line (``thread=``, ``task=``).
Prints the status the endpoint returns; the backend log shows what the running tool does next.
"""

import asyncio
import os
import sys

from google.protobuf import json_format

from livekit.agents.a2a import AgentClient
from livekit.agents.a2a.auth import Credentials

BACKEND = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + "/agents/booking/default/backend"


async def main(thread_id: str, task_id: str) -> None:
    credentials = Credentials(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
    client = AgentClient(BACKEND, credentials.project_token(identity="cancel"))
    try:
        task = await client.cancel(task_id, thread_id=thread_id, reason="review")
    finally:
        await client.aclose()

    returned = json_format.MessageToDict(task)
    print(f"{task_id} {returned.get('status', {}).get('state')} {returned.get('metadata', {})}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
