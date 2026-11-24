"""An agent that ``test_token_mode.py`` runs in token mode.

The test runs ``token_mode_agent.py start`` with ``LIVEKIT_TOKEN`` and no API key or secret.
The job calls the server API through ``ctx.api``, asks for another participant's token, and
writes the results to ``$AGENT_DIR/job.json``.

With ``AGENT_HOLD`` set, the job holds the call instead. When the server closes, the job's
shutdown callback calls the server API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from livekit import api
from livekit.agents import AgentServer, JobContext, cli

AGENT_DIR = Path(os.environ.get("AGENT_DIR", "."))

server = AgentServer(agent_name="token-agent", num_idle_processes=1, load_threshold=1.0)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    if os.environ.get("AGENT_HOLD"):
        await hold(ctx)
        return
    await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.job.room.name))
    seen = {
        "participant_token": await ctx.create_participant_token(identity="avatar", kind="agent")
    }
    (AGENT_DIR / ".job.json").write_text(json.dumps(seen))
    (AGENT_DIR / ".job.json").rename(AGENT_DIR / "job.json")
    ctx.shutdown("done")


async def hold(ctx: JobContext) -> None:
    """Holds the call until the server closes. Then the job's shutdown callback calls the server
    API."""

    async def delete_room() -> None:
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.job.room.name))
        (AGENT_DIR / "shutdown").touch()

    ctx.add_shutdown_callback(delete_room)
    (AGENT_DIR / "job.json").write_text("{}")


if __name__ == "__main__":
    cli.run_app(server)
