"""An ordinary frontend delegates to a separately deployed third-party application."""

import os
from dataclasses import dataclass

from livekit.agents import Agent, AgentServer, SessionState, TextSession, persisted
from livekit.agents.a2a import A2ASessionContext, RemoteAgent
from livekit.agents.llm import ChatContext, FunctionToolCall
from tests.fakes import FakeLLM, last_input, tool_call


def answer(history: ChatContext) -> str | FunctionToolCall:
    last = history.items[-1]
    if last.type == "function_call_output":
        return last.output
    return tool_call("travel", request=last_input(history) or "")


agent = Agent(
    name="desk",
    instructions="Help the person book a flight with the travel agent.",
    llm=FakeLLM(answer),
    tools=[RemoteAgent(os.environ["REMOTE_URL"]).as_tool()],
)
server = AgentServer()


@dataclass
class Conversation:
    session: SessionState = persisted(default_factory=SessionState)


@server.a2a_session(endpoint="desk", description="A travel front desk")
async def text(ctx: A2ASessionContext) -> object:
    app = await ctx.store.load_context(Conversation())
    return await TextSession(agent=agent).run(ctx.message, context=ctx.store.bind(app))
