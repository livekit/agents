import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, CloseEvent, JobContext, cli, llm
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import google, silero

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


# This example shows how to close the agent session when the linked participant disconnects
# or when the worker is shutting down. When closing the session, agent will be interrupted
# and the last agent message will be added to the chat context.


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant.",
            stt="assemblyai/universal-streaming",
            llm="openai/gpt-4.1-mini",
            tts="cartesia/sonic-3",
            tools=[
                EndCallTool(end_instructions="thanks the user for calling and tell them goodbye")
            ],
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="say hello to the user")


class RealtimeAgent(Agent):
    def __init__(self):
        async def on_tool_completed(ev: llm.Toolset.ToolCompletedEvent) -> None:
            # not all realtime models support `generate_reply` inside the tool call
            # use tool reply to say goodbye to the user
            ev.output = ev.output or "thanks the user for calling and tell them goodbye"

        super().__init__(
            instructions="You are a helpful assistant.",
            llm=google.realtime.RealtimeModel(),
            tools=[EndCallTool(end_instructions=None, on_tool_completed=on_tool_completed)],
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="say hello to the user")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
    )

    await session.start(agent=RealtimeAgent(), room=ctx.room)

    @session.on("close")
    def on_close(ev: CloseEvent):
        print(f"Agent Session closed, reason: {ev.reason}")
        print("=" * 20)
        print("Chat History:")
        for item in session.history.items:
            if item.type == "message":
                text = f"{item.role}: {item.text_content.replace('\n', '\\n')}"
                if item.interrupted:
                    text += " (interrupted)"

            elif item.type == "function_call":
                text = f"function_call: {item.name}, arguments: {item.arguments}"

            elif item.type == "function_call_output":
                text = f"{item.name}: '{item.output}'"
                if item.is_error:
                    text += " (error)"

            elif item.type == "agent_handoff":
                text = f"agent_handoff: {item.old_agent_id} -> {item.new_agent_id}"

            else:
                raise ValueError(f"unknown item type: {item.type}")

            print(text)

        print("=" * 20)


if __name__ == "__main__":
    cli.run_app(server)
