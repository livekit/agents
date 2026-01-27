import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, CloseEvent, JobContext, cli
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import google, silero  # noqa: F401

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
            # llm=google.realtime.RealtimeModel(),
            tools=[
                EndCallTool(
                    end_instructions="thanks the user for calling and tell them goodbye",
                    delete_room=True,  # this will disconnect all remote participants, including SIP callers
                )
            ],
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="say hello to the user")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
    )

    await session.start(agent=MyAgent(), room=ctx.room)

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
