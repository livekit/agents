"""HTTP endpoints served alongside an agent.

`server.http` is a FastAPI app, so anything you can do in FastAPI you can do here.
Run it with `python http_endpoints.py dev`, then:

    curl localhost:8081/hello
    curl -X POST localhost:8081/dispatch -H 'content-type: application/json' \
         -d '{"room": "my-room", "identity": "caller"}'
    open localhost:8081/docs
"""

import logging

from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.http import agent_health

logger = logging.getLogger("http-endpoints")

load_dotenv()

server = AgentServer()

server.http.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@server.http.get("/hello")
async def hello() -> dict:
    return {"hello": "world"}


class DispatchRequest(BaseModel):
    room: str
    identity: str


@server.http.post("/dispatch")
async def dispatch(payload: DispatchRequest) -> dict:
    """Take a validated body and act on the agent server.

    A body that does not match DispatchRequest is a 422 before this ever runs.
    """
    # handlers close over `server`: the HTTP server runs in this same process
    logger.info("dispatch requested", extra={"room": payload.room})
    return {"room": payload.room, "identity": payload.identity, "active": len(server.active_jobs)}


async def _verify_admin(x_api_key: str = Header(default="")) -> None:
    if x_api_key != "keep-me-in-an-env-var":
        raise HTTPException(status_code=401, detail="bad api key")


@server.http.get("/admin/jobs", dependencies=[Depends(_verify_admin)])
async def admin_jobs() -> dict:
    """Guard one route with a dependency.

    Prefer this over `add_middleware` for auth: middleware also covers `GET /`, and a
    401 there makes orchestrators restart the process.
    """
    return {"jobs": [job.job.id for job in server.active_jobs]}


@server.http.get("/")
async def health() -> dict:
    """Replace the built-in health check while keeping its checks.

    Defining `GET /` is optional; without it the built-in plain-text one is served.
    """
    reason = agent_health(server)
    return {"ok": reason is None, "reason": reason}


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(llm="openai/gpt-4.1-mini")
    await session.start(
        agent=Agent(instructions="You are a helpful assistant."),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
