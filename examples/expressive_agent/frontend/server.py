"""Token endpoint + static file server for the expressive agent demo frontend.

POST /api/token mints a room token whose agent dispatch carries the session
config (see ../protocol.py — this endpoint passes the blob through untouched;
the agent is the only validator). Everything else is served from this
directory.

Run with LiveKit Cloud credentials in ./.env.local or the environment:

    uv run python server.py
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv

from livekit import api

ROOT = Path(__file__).parent
AGENT_NAME = os.environ.get("EXPRESSIVE_AGENT_NAME", "expressive_agent")
PORT = int(os.environ.get("PORT", "8080"))

load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT.parent / ".env")


async def token(request: web.Request) -> web.Response:
    try:
        config = await request.json()
    except json.JSONDecodeError:
        config = {}

    # the dispatch metadata is the frontend/agent contract; pass it through untouched
    metadata = json.dumps(
        {"expressive": bool(config.get("expressive", True)), "tts": config.get("tts")}
    )

    suffix = secrets.token_hex(4)
    grant = api.VideoGrants(room_join=True, room=f"expressive-demo-{suffix}")
    dispatch = api.RoomAgentDispatch(agent_name=AGENT_NAME, metadata=metadata)
    token = (
        api.AccessToken()
        .with_identity(f"user-{suffix}")
        .with_grants(grant)
        .with_room_config(api.RoomConfiguration(agents=[dispatch]))
    )

    return web.json_response(
        {"server_url": os.environ["LIVEKIT_URL"], "participant_token": token.to_jwt()}
    )


def serve(filename: str):
    # no static-directory route: this directory also holds .env.local
    async def handler(request: web.Request) -> web.FileResponse:
        return web.FileResponse(ROOT / filename)

    return handler


def main() -> None:
    for key in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if not os.environ.get(key):
            raise SystemExit(f"{key} is not set: put it in {ROOT / '.env.local'} or the environment")

    app = web.Application()
    app.router.add_post("/api/token", token)
    app.router.add_get("/", serve("index.html"))
    for filename in ("app.js", "style.css"):
        app.router.add_get(f"/{filename}", serve(filename))
    print(f"expressive agent demo on http://localhost:{PORT}")
    web.run_app(app, port=PORT, print=None)


if __name__ == "__main__":
    main()
