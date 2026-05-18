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


def _pretty(model: str) -> str:
    """`deepgram/nova-3` → `nova-3` for a cleaner spoken form."""
    return model.split("/", 1)[-1] if "/" in model else model


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

    def announce(modality: str, model: str) -> None:
        # Fire-and-forget: say() returns a SpeechHandle that queues
        # behind the current turn (or plays immediately if idle). For
        # TTS swaps the announcement is voiced by the new model, which
        # doubles as an audible confirmation that the swap took.
        session.say(
            f"Switched to {_pretty(model)} for {modality}.",
            allow_interruptions=True,
        )

    @ctx.room.local_participant.register_rpc_method("set_stt_model")
    async def set_stt_model(data: RpcInvocationData) -> str:
        model = parse_value(data.payload, DEFAULT_STT)
        if isinstance(session.stt, inference.STT) and session.stt.model == model:
            return ""
        logger.info("switching STT → %s", model)
        session.stt.update_options(model=model)
        announce("speech-to-text", model)
        return ""

    @ctx.room.local_participant.register_rpc_method("set_llm_model")
    async def set_llm_model(data: RpcInvocationData) -> str:
        model = parse_value(data.payload, DEFAULT_LLM)
        if isinstance(session.llm, inference.LLM) and session.llm.model == model:
            return ""
        logger.info("switching LLM → %s", model)
        session.llm.update_options(model=model)
        announce("the language model", model)
        return ""

    @ctx.room.local_participant.register_rpc_method("set_tts_model")
    async def set_tts_model(data: RpcInvocationData) -> str:
        model = parse_value(data.payload, DEFAULT_TTS)
        if isinstance(session.tts, inference.TTS) and session.tts.model == model:
            return ""
        logger.info("switching TTS → %s", model)
        session.tts.update_options(model=model)
        announce("text-to-speech", model)
        return ""

    await session.start(agent=InferenceAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
