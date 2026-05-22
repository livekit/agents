import json
import logging
from urllib.parse import urlencode

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    inference,
)
from livekit.rtc import RpcInvocationData

logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)

load_dotenv()

DEFAULT_STT = "deepgram/nova-3"
DEFAULT_LLM = "openai/gpt-4.1-mini"
DEFAULT_TTS = "cartesia/sonic-3"

# Default starter prompt. Keep in sync with the `set_system_prompt`
# control's `default` in examples/playground.yaml — the UI seeds the
# textarea with the same string so the first session before any edit
# matches what the user sees.
INSTRUCTIONS = (
    "You're a friendly agent in the LiveKit Playground. The person "
    "talking to you is prototyping their own voice agent — they can "
    "edit this prompt in the side panel and swap the STT / LLM / TTS "
    "models live. Keep replies short, natural, and conversational. "
    "If asked which models you're using, answer honestly."
)

_SWAP_PROMPT = (
    "The user just switched the {modality} model to '{model}'. "
    "Acknowledge it in one short, natural sentence — say the model's "
    "name like a brand (e.g. 'Deepgram Nova 3', not 'deepgram slash "
    "nova dash three'). Skip hyphens, slashes, version dots, and any "
    "abbreviations that aren't pronounceable. Don't ask a follow-up."
)


class InferenceAgent(Agent):
    def __init__(self, instructions: str = INSTRUCTIONS) -> None:
        super().__init__(instructions=instructions)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT(model=DEFAULT_STT),
        llm=inference.LLM(model=DEFAULT_LLM),
        tts=inference.TTS(model=DEFAULT_TTS),
    )

    def parse_value(payload: str, fallback: str) -> str:
        try:
            v = json.loads(payload).get("value")
            return v if isinstance(v, str) and v else fallback
        except Exception:
            return fallback

    agent = InferenceAgent()
    await session.start(agent=agent, room=ctx.room)

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

    @ctx.room.local_participant.register_rpc_method("open_in_builder")
    async def open_in_builder(data: RpcInvocationData) -> str:
        # Build the Cloud Builder deep-link agent-side so the
        # frontend doesn't have to know the URL schema. `p_` is a
        # placeholder project_id — Cloud routes the user through
        # login if needed and preserves the params on redirect.
        params = {
            "modelMode": "pipeline",
            "instructions": agent.instructions or "",
            "llm": session.llm.model if isinstance(session.llm, inference.LLM) else DEFAULT_LLM,
            "stt": session.stt.model if isinstance(session.stt, inference.STT) else DEFAULT_STT,
            "tts": session.tts.model if isinstance(session.tts, inference.TTS) else DEFAULT_TTS,
        }
        return f"https://cloud.livekit.io/projects/p_/agents/builder/new?{urlencode(params)}"

    @ctx.room.local_participant.register_rpc_method("set_system_prompt")
    async def set_system_prompt(data: RpcInvocationData) -> str:
        # The UI fires this on every keystroke (debounced client-side
        # by the textarea's edit→commit boundary), so dedupe against
        # the current value before touching the agent. update_instructions
        # is cheap but it logs.
        prompt = parse_value(data.payload, "")
        if not prompt:
            return ""
        if agent.instructions == prompt:
            return ""
        logger.info("system prompt updated (%d chars)", len(prompt))
        await agent.update_instructions(prompt)
        return ""


if __name__ == "__main__":
    cli.run_app(server)
