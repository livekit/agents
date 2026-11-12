"""Scripted application served by a real AgentServer in a separate Python process.

Only model choice and the application service are scripted. Agent execution, tools,
approval, partition_key leases, persistence and A2A streaming use the production implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import timedelta

import httpx
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from livekit.agents import Agent, AgentServer, InputRequired, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import ChatContext
from livekit.agents.storage import AgentDBStorage
from tests_new.fakes import FakeLLM, Reply, last_input, tool_call

from .resources import REQUEST_SECONDS, ServiceCall

BOOK = {"terms": "slot-9 $100"}


def answer(chat_ctx: ChatContext) -> Reply:
    """The model's reply from the request alone: workers share no per-conversation state."""
    items = [item for item in chat_ctx.items if item.type != "agent_handoff"]
    last, text = items[-1], last_input(chat_ctx) or ""
    if last.type == "function_call_output":
        if last.name == "book" and "not run yet:" in text:
            approved = "answered 'yes'" in text
            return tool_call("book", call_id="approved", **BOOK) if approved else "declined"
        return "completed: " + text
    if text == "recall":
        return next(i.text_content or "" for i in items if i.type == "message" and i.role == "user")
    replies: dict[str, Reply] = {
        "block": tool_call("lookup", call_id="lookup"),
        "approve": tool_call("book", call_id="ask", **BOOK),
        "need-email": tool_call("email", call_id="email"),
        "yes": tool_call("book", call_id="approved", **BOOK),
        "no": "declined",
        "alice@example.com": "completed: alice@example.com",
    }
    return replies.get(text, "reply: " + text)


def make_app(worker: str, service_url: str, port: int) -> AgentServer:
    storage = AgentDBStorage(
        holder=worker, lease_ttl=timedelta(seconds=2), renew_interval=timedelta(seconds=0.5)
    )
    server = AgentServer(
        storage=storage,
        agent_name="e2e",
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
        listeners={"public": {"port": port}, "health": {"port": 0}},
    )
    agent = Agent(
        context_type=ServiceCall,
        name="e2e",
        instructions="Follow the scripted model.",
        llm=FakeLLM(answer),
    )

    async def service(call: ServiceCall) -> str:
        async with httpx.AsyncClient(timeout=REQUEST_SECONDS, trust_env=False) as http:
            response = await http.post(service_url + "/call", json=call.model_dump())
            response.raise_for_status()
            return response.text

    @agent.tool
    async def lookup(ctx: RunContext[ServiceCall]) -> str:
        """Wait for the external lookup service."""
        return await service(ctx.context.model_copy(update={"operation": "wait"}))

    @agent.tool(needs_approval=lambda terms: f"Book {terms}? Yes or no?")
    async def book(ctx: RunContext[ServiceCall], terms: str) -> str:
        """Book the exact quoted terms."""
        return await service(ctx.context.model_copy(update={"operation": "book", "terms": terms}))

    @agent.tool
    async def email() -> InputRequired:
        """Ask for the confirmation email."""
        return InputRequired("Which email?", verbatim=True)

    @server.http.get("/ready", public=True)
    async def ready(request: Request) -> PlainTextResponse:
        return PlainTextResponse(worker)

    async def handler(ctx: A2ASessionContext) -> object:
        await service(
            ServiceCall(
                worker=worker,
                thread=ctx.thread.id,
                operation="turn",
                terms=ctx.message.text_content or "",
            )
        )
        return await agent.run(
            ctx.message,
            message_history=ctx.message_history,
            context=ServiceCall(worker=worker, thread=ctx.thread.id, operation=""),
        )

    server.a2a_session(endpoint="desk", description="E2E scripted desk")(handler)
    return server


if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s %(message)s %(error)s", defaults={"error": ""})
    )
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--service", required=True)
    args = parser.parse_args()
    server = make_app(args.worker, args.service, args.port)
    # the worker serves its routes and text doors on its own loopback listener
    asyncio.run(server._run(devmode=True, unregistered=True))
