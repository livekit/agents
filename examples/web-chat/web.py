"""The chat page and its tokens, on http://localhost:8000; run it beside the agent."""

import os
import re
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from livekit.agents.a2a import AgentThreadGrant, Credentials
from livekit.agents.storage import AgentDBStorage

load_dotenv()

KEY, SECRET = os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
AGENT = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + "/agents/web-chat/_"
GRANT_TTL = 15 * 60  # seconds; the page asks again once a token expires

credentials = Credentials(KEY, SECRET)
threads = AgentDBStorage(api_key=KEY, api_secret=SECRET).threads


# tokens for one browser, whose id is random and stays in its localStorage: without `thread`,
# to list and start the browser's threads; with it, to read and continue that one thread, once
# it is checked to be the browser's
async def token(request: Request) -> Response:
    browser = request.query_params.get("browser", "")
    if not re.fullmatch(r"[0-9a-f-]{36}", browser):
        return JSONResponse({"error": "browser must be a UUID"}, status_code=400)
    scope = f"browser:{browser}"
    thread = request.query_params.get("thread")
    if thread is None:
        grant = AgentThreadGrant(scope=scope, list=True, create=True, write=True)
    elif re.fullmatch(r"AT_\w+", thread) and (await threads.get_thread(thread)).scope == scope:
        grant = AgentThreadGrant(thread_ids=[thread], read=True, write=True)
    else:
        return JSONResponse({"error": "not this browser's thread"}, status_code=403)
    token = credentials.thread_grant_token(grant, ttl_seconds=GRANT_TTL)
    return JSONResponse({"token": token, "ttl": GRANT_TTL, "scope": scope, "agent": AGENT})


app = Starlette(
    routes=[
        Route("/token", token),
        Mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True)),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
