import json
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    inference,
)
from livekit.plugins import silero
from livekit.rtc import RpcInvocationData

logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)

load_dotenv()

# Match the defaults declared in examples/playground.yaml so the agent
# boots with the same models the playground initially shows in the
# control pills.
DEFAULT_STT = "deepgram/nova-3"
DEFAULT_LLM = "openai/gpt-4o-mini"
DEFAULT_TTS = "cartesia/sonic-2"

INSTRUCTIONS = (
    "You're a friendly demo agent showcasing LiveKit Inference. "
    "Keep replies short, natural, and conversational. If asked which "
    "models you're using, answer honestly — they swap live as the user "
    "picks new ones in the playground."
)


class InferenceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT(model=DEFAULT_STT),
        llm=inference.LLM(model=DEFAULT_LLM),
        tts=inference.TTS(model=DEFAULT_TTS),
        vad=silero.VAD.load(),
    )

    def parse_value(payload: str, fallback: str) -> str:
        try:
            v = json.loads(payload).get("value")
            return v if isinstance(v, str) and v else fallback
        except Exception:
            return fallback

    @ctx.room.local_participant.register_rpc_method("set_stt_model")
    async def set_stt_model(data: RpcInvocationData) -> str:
        model = parse_value(data.payload, DEFAULT_STT)
        logger.info("switching STT → %s", model)
        session.stt.update_options(model=model)
        return ""

    @ctx.room.local_participant.register_rpc_method("set_llm_model")
    async def set_llm_model(data: RpcInvocationData) -> str:
        model = parse_value(data.payload, DEFAULT_LLM)
        logger.info("switching LLM → %s", model)
        # inference.LLM has no update_options; _opts.model is read on
        # every chat() call so this swaps in for the next reply.
        session.llm._opts.model = model
        return ""

    @ctx.room.local_participant.register_rpc_method("set_tts_model")
    async def set_tts_model(data: RpcInvocationData) -> str:
        model = parse_value(data.payload, DEFAULT_TTS)
        logger.info("switching TTS → %s", model)
        session.tts.update_options(model=model)
        return ""

    await session.start(agent=InferenceAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
