"""The app's backend, on http://localhost:8001: the page's tokens and its threads.

It holds the project key; the agent itself runs elsewhere and is reached through LiveKit Cloud.
"""

import os
import re

import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from livekit.agents.a2a import AgentThreadGrant, Credentials
from livekit.agents.storage import AgentDBStorage, Thread

load_dotenv()

KEY, SECRET = os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
# the agent's endpoints on LiveKit Cloud, under the deployment `lk agent dev` registered it with
DEPLOYMENT = os.environ.get("LIVEKIT_AGENT_DEPLOYMENT", "_")
AGENT = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + f"/agents/web-chat/{DEPLOYMENT}"
GRANT_TTL = 15 * 60  # seconds; the page asks again once a token expires

credentials = Credentials(KEY, SECRET)
storage = AgentDBStorage(api_key=KEY, api_secret=SECRET)


# the browser's scope, from the random id the page keeps in its localStorage
def scope_of(request: Request) -> str | None:
    browser = request.query_params.get("browser", "")
    return f"browser:{browser}" if re.fullmatch(r"[0-9a-f-]{36}", browser) else None


# whether `thread` is one of the browser's threads
async def owns(scope: str, thread: str) -> bool:
    if not re.fullmatch(r"AT_\w+", thread):
        return False
    return (await storage.threads.get_thread(thread)).scope == scope


# a token to send: without `thread`, to start a thread in the browser's scope; with it, to
# continue that thread
async def token(request: Request) -> Response:
    scope, thread = scope_of(request), request.query_params.get("thread")
    if scope is None:
        return Response(status_code=400)
    if thread is None:
        grant = AgentThreadGrant(scope=scope, create=True, write=True)
    elif await owns(scope, thread):
        grant = AgentThreadGrant(thread_ids=[thread], read=True, write=True)
    else:
        return Response(status_code=403)
    token = credentials.thread_grant_token(grant, ttl_seconds=GRANT_TTL)
    return JSONResponse({"token": token, "ttl": GRANT_TTL, "agent": AGENT})


# the browser's threads, newest first
async def list_threads(request: Request) -> Response:
    scope = scope_of(request)
    if scope is None:
        return Response(status_code=400)
    page = await storage.threads.list_threads(scope=scope, limit=50)
    return JSONResponse([{"id": t.thread_id, "created_at": t.created_at} for t in page.threads])


# one thread's user and assistant messages, in order
async def thread_messages(request: Request) -> Response:
    scope, thread = scope_of(request), request.path_params["thread"]
    if scope is None or not await owns(scope, thread):
        return Response(status_code=403)
    messages, after = [], None
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
    ],
    # the page is served from its own origin
    middleware=[Middleware(CORSMiddleware, allow_origins=["http://localhost:8000"])],
)

if __name__ == "__main__":
    uvicorn.run(app, port=8001)
