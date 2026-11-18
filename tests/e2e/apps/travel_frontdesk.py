"""An ordinary frontend delegates to a separately deployed third-party application."""

import json
import os
from dataclasses import dataclass

from livekit.agents import (
    Agent,
    AgentServer,
    DelegationOptions,
    SessionState,
    TextSession,
    persisted,
)
from livekit.agents.a2a import A2ASessionContext, RemoteAgent
from livekit.agents.llm import ChatContext, FunctionToolCall
from tests.fakes import FakeLLM, last_input, tool_call


def answer(history: ChatContext) -> str | FunctionToolCall:
    last = history.items[-1]
    if last.type == "function_call_output":
        return last.output
    if last_input(history) == "Why do you need the date?":
        return "The travel agent needs it to check availability."
    return tool_call("travel", request=last_input(history) or "")


def route_response(history: ChatContext) -> FunctionToolCall:
    message = history.items[-1]
    assert message.type == "message" and message.text_content is not None
    prompt = json.loads(message.text_content)
    if "proposed_call" in prompt:
        assert prompt["proposed_call"] == {"request": "Friday"}
        assert prompt["selected_task"] is None
        [candidate] = prompt["candidates"]
        [response] = candidate["responses"]
        assert response["question"] == "Which day?"
        assert response["held_response"] == "Friday"
        return tool_call(
            "route_remote_call",
            action="respond",
            candidate=candidate["index"],
            response=response["index"],
        )
    assert prompt["latest_message"] in ("Why do you need the date?", "Friday")
    [candidate] = prompt["candidates"]
    assert candidate["question"] == "Which day?"
    return tool_call(
        "route_input_response",
        candidates=[candidate["index"]] if prompt["latest_message"] == "Friday" else [],
        unclear=False,
    )


agent = Agent(
    name="desk",
    instructions="Help the person book a flight with the travel agent.",
    llm=FakeLLM(answer),
    tools=[RemoteAgent(os.environ["REMOTE_URL"]).as_tool()],
)
server = AgentServer()
delegation = DelegationOptions(routing_model=FakeLLM(route_response))


@dataclass
class Conversation:
    session: SessionState = persisted(default_factory=SessionState)


@server.a2a_session(endpoint="desk", description="A travel front desk")
async def text(ctx: A2ASessionContext) -> object:
    app = await ctx.store.load_context(Conversation())
    return await TextSession(agent=agent, delegation=delegation).run(
        ctx.message, context=ctx.store.bind(app)
    )
