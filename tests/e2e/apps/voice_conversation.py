"""The ordinary voice application used to verify the shared room caller."""

from livekit.agents import AgentServer, VoiceSession
from livekit.agents.job import JobContext

from ..framework.media import ToneSTT, ToneTTS
from .conversation_agent import agent

server = AgentServer()


@server.rtc_session()
async def voice(ctx: JobContext) -> None:
    session = VoiceSession(
        stt=ToneSTT(),
        tts=ToneTTS(),
        vad=None,
        turn_handling={"turn_detection": "stt"},
        aec_warmup_duration=0,
    )
    await session.start(agent=agent(), room=ctx.room)
