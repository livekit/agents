"""An agent run as a LiveKit Cloud pod runs it, by ``test_cloud_hosting.py``: ``cloud_pod.py start``.

Its job writes what it sees to ``$POD_DIR/job.json``, then holds the call until ``$POD_DIR/hangup``
exists. It asks for a load policy a Cloud pod must ignore: always full.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from livekit.agents import AgentServer, JobContext, cli, inference, utils

POD_DIR = Path(os.environ.get("POD_DIR", "."))

server = AgentServer(
    agent_name="cloud-agent",
    listeners={"health": {"host": "127.0.0.1", "port": int(os.environ.get("POD_HEALTH_PORT", 0))}},
    num_idle_processes=1,
    load_threshold=0.0,
    _load_fnc=lambda: 1.0,
)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    seen = {
        "pid": os.getpid(),
        "env": {name: os.environ.get(name) for name in os.environ if name.startswith("LIVEKIT_")},
        "inference_headers": inference.get_inference_headers(),
        "inference_url": inference.get_default_inference_url(),
        "hosted": utils.is_hosted(),
    }
    (POD_DIR / ".job.json").write_text(json.dumps(seen))
    (POD_DIR / ".job.json").rename(POD_DIR / "job.json")
    while not (POD_DIR / "hangup").exists():
        await asyncio.sleep(0.05)
    ctx.shutdown("hung up")


if __name__ == "__main__":
    cli.run_app(server)
