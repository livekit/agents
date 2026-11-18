"""The chat page and its backend, on http://localhost:8000.

It holds the project key, so the browser never does. The agent runs elsewhere and is
reached through LiveKit Cloud.
"""

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
from livekit.agents.storage import AgentDBStorage, Thread

load_dotenv()

KEY = os.environ["LIVEKIT_API_KEY"]
SECRET = os.environ["LIVEKIT_API_SECRET"]

# The agent's endpoints on LiveKit Cloud, under the deployment `lk agent dev` registered.
DEPLOYMENT = os.environ.get("LIVEKIT_AGENT_DEPLOYMENT", "_")
AGENT = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + f"/agents/web-chat/{DEPLOYMENT}"

TOKEN_TTL = 15 * 60  # seconds

credentials = Credentials(KEY, SECRET)
storage = AgentDBStorage(api_key=KEY, api_secret=SECRET)


# --- Who is asking ---------------------------------------------------------------------------


def scope_of(request: Request) -> str | None:
    """The browser's scope, from the random id the page keeps in localStorage."""
    browser = request.query_params.get("browser", "")
    if not re.fullmatch(r"[0-9a-f-]{36}", browser):
        return None
    return f"browser:{browser}"


async def owns(scope: str, thread: str) -> bool:
    """Whether the thread belongs to the browser."""
    if not re.fullmatch(r"AT_\w+", thread):
        return False
    info = await storage.threads.get_thread(thread)
    return info.scope == scope


# --- Routes ----------------------------------------------------------------------------------


async def token(request: Request) -> Response:
    """A token to message the agent: to start a new thread, or to continue one."""
    scope = scope_of(request)
    thread = request.query_params.get("thread")
    if scope is None:
        return Response(status_code=400)

    if thread is None:
        grant = AgentThreadGrant(scope=scope, create=True, write=True)
    elif await owns(scope, thread):
        grant = AgentThreadGrant(thread_ids=[thread], read=True, write=True)
    else:
        return Response(status_code=403)

    token = credentials.thread_grant_token(grant, ttl_seconds=TOKEN_TTL)
    return JSONResponse({"token": token, "ttl": TOKEN_TTL, "agent": AGENT})


async def list_threads(request: Request) -> Response:
    """The browser's threads, newest first."""
    scope = scope_of(request)
    if scope is None:
        return Response(status_code=400)

    page = await storage.threads.list_threads(scope=scope, limit=50)
    return JSONResponse([{"id": t.thread_id, "created_at": t.created_at} for t in page.threads])


async def thread_messages(request: Request) -> Response:
    """A thread's user and assistant messages, in order."""
    scope = scope_of(request)
    thread = request.path_params["thread"]
    if scope is None or not await owns(scope, thread):
        return Response(status_code=403)

    messages: list[dict[str, str]] = []
    after: str | None = None
    while after != "":
        page = await Thread(storage, thread).list_items(after=after)
        for entry in page.items:
            item = entry.item
            if item.type == "message" and item.role in ("user", "assistant"):
                messages.append({"role": item.role, "text": item.text_content or ""})
        after = page.after

    return JSONResponse(messages)


app = Starlette(
    routes=[
        Route("/token", token),
        Route("/threads", list_threads),
        Route("/threads/{thread}", thread_messages),
        Mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True)),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
