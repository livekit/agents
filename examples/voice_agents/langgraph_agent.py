import logging
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, langchain, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("basic-agent")

load_dotenv()

# this example demonstrates adding Voice to a LangGraph graph by using our
# adapter.
# In this example, instructions and tool calls are handled in LangGraph, while
# voice orchestration (turns, interruptions, etc) are handled by Agents framework


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def create_graph() -> StateGraph:
    openai_llm = init_chat_model(
        model="openai:gpt-4o",
    )

    def chatbot_node(state: State):
        return {"messages": [openai_llm.invoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    return builder.compile()


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    graph = create_graph()

    agent = Agent(
        instructions="",
        llm=langchain.LLMAdapter(graph),
    )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=deepgram.TTS(),
        # use LiveKit's turn detection model
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # to use Krisp background voice cancellation, install livekit-plugins-noise-cancellation
            # and `from livekit.plugins import noise_cancellation`
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()
    await session.generate_reply(instructions="ask the user how they are doing?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
