"""Scripted application served by a real AgentServer in a separate Python process.

Only model choice and the application service are scripted. Agent execution, tools,
approval, lane leases, persistence and A2A streaming use the production implementation.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
from datetime import timedelta

import httpx
import uvicorn
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from livekit.agents import Agent, AgentServer, InputRequired, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import ChatContext, FunctionToolCall, LLMStream, Tool
from livekit.agents.storage import AgentDBStorage
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from tests.fake_llm import FakeLLM, FakeLLMResponse

from .resources import DEBOUNCE_SECONDS, REQUEST_SECONDS, ServiceCall


class ScriptedLLM(FakeLLM):
    """Request-local responses; no mutable per-conversation state in the shared model."""

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: object,
    ) -> LLMStream:
        items = [item for item in chat_ctx.items if item.type != "agent_handoff"]
        last = items[-1]
        text = ""
        calls: list[FunctionToolCall] = []
        if last.type == "function_call_output":
            key = last.output
            if last.name == "book" and "not run yet:" in key:
                if "answered 'yes'" in key:
                    calls = [
                        FunctionToolCall(
                            name="book", arguments='{"terms":"slot-9 $100"}', call_id="approved"
                        )
                    ]
                else:
                    text = "declined"
            else:
                text = "completed: " + key
        elif last.type == "message":
            key = last.text_content or ""
            if key == "block":
                calls = [FunctionToolCall(name="lookup", arguments="{}", call_id="lookup")]
            elif key == "approve":
                calls = [
                    FunctionToolCall(
                        name="book", arguments='{"terms":"slot-9 $100"}', call_id="ask"
                    )
                ]
            elif key == "need-email":
                calls = [FunctionToolCall(name="email", arguments="{}", call_id="email")]
            elif key == "yes":
                calls = [
                    FunctionToolCall(
                        name="book", arguments='{"terms":"slot-9 $100"}', call_id="approved"
                    )
                ]
            elif key == "no":
                text = "declined"
            elif key == "alice@example.com":
                text = "completed: " + key
            elif key == "recall":
                text = next(
                    item.text_content or ""
                    for item in items
                    if item.type == "message" and item.role == "user"
                )
            else:
                text = "reply: " + key
        else:
            raise ValueError(f"unsupported scripted model input: {last.type}")
        response = FakeLLMResponse(input=key, content=text, tool_calls=calls, ttft=0, duration=0)
        return FakeLLM(fake_responses=[response]).chat(
            chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
        )


def make_app(worker: str, service_url: str) -> AgentServer:
    storage = AgentDBStorage(
        holder=worker, lease_ttl=timedelta(seconds=2), renew_interval=timedelta(seconds=0.5)
    )
    server = AgentServer(
        storage=storage,
        agent_name="e2e",
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
    )
    agent = Agent[ServiceCall](
        name="e2e", instructions="Follow the scripted model.", llm=ScriptedLLM()
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

    @server.http.get("/ready")
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
    server.a2a_session(
        endpoint="inbox",
        description="E2E inbox",
        debounce=timedelta(seconds=DEBOUNCE_SECONDS),
    )(handler)
    return server


if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s %(message)s %(error)s", defaults={"error": ""})
    )
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--service", required=True)
    args = parser.parse_args()
    server = make_app(args.worker, args.service)
    sock = socket.socket(fileno=args.fd)
    uvicorn.Server(
        uvicorn.Config(server.http.app, log_level="warning", timeout_graceful_shutdown=1)
    ).run(sockets=[sock])
