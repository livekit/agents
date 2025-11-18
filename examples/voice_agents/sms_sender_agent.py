import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.beta.tools.sms import SMSToolConfig, create_sms_tool
from livekit.plugins import silero

logger = logging.getLogger("sms-agent")

load_dotenv()

RANDOM_MESSAGES = [
    "Here's your daily reminder that you're awesome!",
    "Hope your day is going great, consider this a little boost!",
    "Just popping in to say hi from your friendly bot",
    "This is a quick systems check: all vibes appear positive.",
]

IS_SIP_SESSION = True # Enables auto-detection of caller's number in SIP session
# Disable it for non-SIP sessions (e.g. console runs)

sms_tool = create_sms_tool(
    SMSToolConfig(
        name="send_playful_sms",
        description=(
            IS_SIP_SESSION and "Send a playful SMS to the caller's phone number. "
            or "Send a playful SMS to the provided `to` phone number. "
            "Always include the exact message text you want to deliver."
        ),
        auto_detect_caller=IS_SIP_SESSION,
        execution_message="One moment while I send that SMS.",
    )
)

instructions = (
    "Keep it minimal. Immediately greet the user and tell them you are ready to send a light-hearted SMS. "
    "Ask once for their mobile number with country code. "
    "Do not list options or chat idly — just confirm you are waiting for the number. "
    "After they provide a valid number (7-15 digits), call send_playful_sms with that number as `to` and choose any one of these messages: "
    f"{'; '.join(RANDOM_MESSAGES)} "
    "Send the SMS silently and then simply say “Done. Check your phone.”"
)

if IS_SIP_SESSION:
    instructions = (
        "Keep it minimal. Immediately greet the user and tell them you are ready to send a light-hearted SMS. "
        "Ask to choose a message to send"
        f"{'; '.join(RANDOM_MESSAGES)} "
        "Send the SMS silently and then simply say “Done. Check your phone.”"
    )


class SMSAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=instructions, tools=[sms_tool])

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=
                IS_SIP_SESSION
                and "Warmly greet the user and ask to choose a message to send"
                or "Warmly greet the user and ask for their mobile number (skip if you don't have to field in sms tool)"
            )

server = AgentServer()

@server.rtc_session(agent_name="sms-sender")
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4o-mini",
        tts="cartesia/sonic-3:a167e0f3-df7e-4d52-a9c3-f949145efdab",
        vad=silero.VAD.load(),
    )

    await session.start(agent=SMSAgent(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
