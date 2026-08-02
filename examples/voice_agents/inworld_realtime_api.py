from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    function_tool,
    llm,
    room_io,
)
from livekit.plugins.inworld.realtime import RealtimeModel

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Jessica, a helpful assistant",
            llm=RealtimeModel(
                model="google-ai-studio/gemini-3.1-flash-lite",
                voice="Ashley",
                tts_model="inworld-tts-2",
                stt_model="inworld/inworld-stt-1",
                modalities=["audio"],
                provider_data={"auto_tool_response": False},
            ),
        )

    async def on_enter(self):
        chat_history = [
            {
                "role": "user",
                "content": "Hello. I'm just picking up.",
            },
        ]
        # Google models require a user item to generate a response, so pass it through chat_ctx.
        chat_ctx = llm.ChatContext.empty()
        for item in chat_history:
            chat_ctx.add_message(role=item["role"], content=item["content"])

        self.session.generate_reply(
            instructions="introduce yourself very briefly and ask about the user's day",
            chat_ctx=chat_ctx,
        )

    @function_tool
    async def get_weather(self, city: str):
        """Get the weather for a given city"""
        return f"The weather in {city} is sunny and 70 degrees"


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession()

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            text_output=room_io.TextOutputOptions(transcription_speed_factor=1.5),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
