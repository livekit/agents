import os
import re
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from livekit.agents import Agent, AgentServer, RunResult, TextSession, cli, inference
from livekit.agents.a2a import A2ASessionContext, AgentThreadGrant, Credentials
from livekit.agents.storage import AgentDBStorage

load_dotenv()

STATIC = Path(__file__).resolve().parent / "static"  # the page: index.html, style.css, app.js
GRANT_TTL = 15 * 60  # seconds; the page asks for a new grant once it expires

assistant = Agent(
    instructions="You are a friendly, helpful assistant in a web chat. Answer clearly and "
    "briefly, in plain text.",
    llm=inference.LLM(model="google/gemma-4-31b-it"),
)

server = AgentServer(storage=AgentDBStorage(), agent_name="web-chat")


# the text door: one turn per message, the history kept in the message's thread
@server.a2a_session(endpoint="chat", description="A general chat assistant.")
async def chat(ctx: A2ASessionContext) -> RunResult[str]:
    turn = await TextSession(agent=assistant).run(ctx.message, context=ctx.store.bind())
    return turn.result


@server.http.get("/", public=True)
async def page(request: Request) -> Response:
    return FileResponse(STATIC / "index.html")


@server.http.get("/static/{path:path}", public=True)
async def static(request: Request) -> Response:
    file = (STATIC / request.path_params["path"]).resolve()
    if not file.is_relative_to(STATIC) or not file.is_file():
        return Response(status_code=404)
    return FileResponse(file)


# a thread grant for one browser: it lists, starts, reads and continues the threads of its own
# scope, and no other; the browser's id is random and stays in its localStorage
@server.http.get("/token", public=True)
async def token(request: Request) -> Response:
    browser = request.query_params.get("browser", "")
    if not re.fullmatch(r"[0-9a-f-]{36}", browser):
        return JSONResponse({"error": "browser must be a UUID"}, status_code=400)
    grant = AgentThreadGrant(
        scope=f"browser:{browser}", list=True, create=True, read=True, write=True
    )
    credentials = Credentials(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
    return JSONResponse(
        {
            "token": credentials.thread_grant_token(grant, ttl_seconds=GRANT_TTL),
            "ttl": GRANT_TTL,
            "scope": grant.scope,
            "api": public_api(os.environ["LIVEKIT_URL"]),
        }
    )


def public_api(project_url: str) -> str:
    """The LiveKit public API, where the browser reads its threads."""
    host = urlparse(project_url).hostname or ""
    if host.endswith(".staging.livekit.cloud"):
        return "https://api.staging.livekit.cloud"
    if host.endswith(".livekit.cloud"):
        return "https://api.livekit.cloud"
    return project_url.replace("ws", "http", 1)  # a self-hosted server


if __name__ == "__main__":
    cli.run_app(server)
