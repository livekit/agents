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

DEFAULT_STT = "deepgram/nova-3"
DEFAULT_LLM = "openai/gpt-4o-mini"
DEFAULT_TTS = "cartesia/sonic-2"

INSTRUCTIONS = (
    "You're a friendly demo agent showcasing LiveKit Inference. "
    "Keep replies short, natural, and conversational. If asked which "
    "models you're using, answer honestly — they swap live as the user "
    "picks new ones in the playground."
)

_SWAP_PROMPT = (
    "The user just switched the {modality} model to '{model}'. "
    "Acknowledge it in one short, natural sentence — say the model's "
    "name like a brand (e.g. 'Deepgram Nova 3', not 'deepgram slash "
    "nova dash three'). Skip hyphens, slashes, version dots, and any "
    "abbreviations that aren't pronounceable. Don't ask a follow-up."
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

    await session.start(agent=InferenceAgent(), room=ctx.room)

    @ctx.room.local_participant.register_rpc_method("set_stt_model")
    async def set_stt_model(data: RpcInvocationData) -> str:
        model = parse_value(data.payload, DEFAULT_STT)
        if isinstance(session.stt, inference.STT) and session.stt.model == model:
            return ""
        logger.info("switching STT → %s", model)
        session.stt.update_options(model=model)
        session.generate_reply(
            instructions=_SWAP_PROMPT.format(modality="speech-to-text", model=model)
        )
        return ""

    @ctx.room.local_participant.register_rpc_method("set_llm_model")
    async def set_llm_model(data: RpcInvocationData) -> str:
        model = parse_value(data.payload, DEFAULT_LLM)
        if isinstance(session.llm, inference.LLM) and session.llm.model == model:
            return ""
        logger.info("switching LLM → %s", model)
        session.llm.update_options(model=model)
        session.generate_reply(instructions=_SWAP_PROMPT.format(modality="language", model=model))
        return ""

    @ctx.room.local_participant.register_rpc_method("set_tts_model")
    async def set_tts_model(data: RpcInvocationData) -> str:
        model = parse_value(data.payload, DEFAULT_TTS)
        if isinstance(session.tts, inference.TTS) and session.tts.model == model:
            return ""
        logger.info("switching TTS → %s", model)
        session.tts.update_options(model=model)
        session.generate_reply(
            instructions=_SWAP_PROMPT.format(modality="text-to-speech", model=model)
        )
        return ""


if __name__ == "__main__":
    cli.run_app(server)
