"""Lists a thread's task rows through the booking office's ``tasks/list``.

``uv run python tasks.py <thread_id>``, with the worker's environment; the thread id is the
``thread=`` of a ``backend turn`` log line. The endpoint returns every lane's rows, oldest
first, so the travel desk's turns show between the booking office's.
"""

import asyncio
import json
import os
import sys

from livekit.agents.a2a import AgentClient
from livekit.agents.a2a.auth import Credentials

BACKEND = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + "/agents/booking/default/backend"


async def main(thread_id: str) -> None:
    credentials = Credentials(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
    client = AgentClient(BACKEND, credentials.project_token(identity="review"))
    try:
        await client.connect()
        # tasks/list has no client method; the private _rpc is fine in a diagnostic script
        result = await client._rpc("ListTasks", {"contextId": thread_id, "pageSize": 200})
    finally:
        await client.aclose()

    tasks = result.get("tasks")
    for task in tasks if isinstance(tasks, list) else []:
        assert isinstance(task, dict)
        status = task.get("status")
        state = status.get("state") if isinstance(status, dict) else None
        metadata = json.dumps(task.get("metadata") or {})
        print(f"{task.get('id')} {state} {metadata}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
