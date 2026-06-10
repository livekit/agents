import json
import logging
import os
from collections.abc import AsyncGenerator
from urllib.parse import urlencode

from dotenv import load_dotenv

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentServer,
    AgentSession,
    APIError,
    JobContext,
    ModelSettings,
    cli,
    inference,
    llm,
    tokenize,
)
from livekit.agents.voice import CONVERSATIONAL_EXPRESSIVENESS_PRESET
from livekit.plugins import openai, silero
from livekit.rtc import RpcInvocationData

logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)

load_dotenv()

DEFAULT_STT = "deepgram/nova-3"
DEFAULT_LLM = "gemma-4-31b-it"
DEFAULT_TTS = "inworld/inworld-tts-2"

GEMMA_BASE_URL = os.environ["GEMMA_BASE_URL"]
GEMMA_API_KEY = os.environ["GEMMA_API_KEY"]
GEMMA_MODEL = "gemma-4-31b-it"

# Seed model for the Inference backend — only used once a non-Gemma model
# is picked, but inference.LLM needs a valid model at construction.
FALLBACK_INFERENCE_LLM = "openai/gpt-4.1-mini"

# Default starter prompt. Keep in sync with the `set_system_prompt`
# control's `default` in examples/playground.yaml — the UI seeds the
# textarea with the same string so the first session before any edit
# matches what the user sees.
INSTRUCTIONS = (
    "You're a friendly agent in the LiveKit Playground. The person "
    "talking to you is prototyping their own voice agent — they can "
    "edit this prompt in the side panel and swap the STT / LLM / TTS "
    "models live. Keep replies short, natural, and conversational, and "
    "be expressive so they can hear what the selected voice can do. "
    "If the conversation lulls or they're not sure what to try, offer "
    "to tell them a short joke — and if they say yes, deliver it with "
    "good comic timing. If asked which models you're using, answer honestly."
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
        # The session's default LLM is the Gemma openai.LLM (see entrypoint).
        # Every other model in playground.yaml is on LiveKit Inference, which
        # update_options() can't reach from the Gemma plugin — so we keep a
        # separate Inference backend and route to it from llm_node.
        self._inference_llm = inference.LLM(model=FALLBACK_INFERENCE_LLM)
        self.llm_model = DEFAULT_LLM

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the user with excitement, and ask how their day has been "
            "with curiosity."
        )

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[llm.ChatChunk | str, None]:
        # Gemma → the session's default LLM; anything else → Inference.
        active = self.session.llm if self.llm_model == GEMMA_MODEL else self._inference_llm
        logger.info("llm_node → llm_model=%s backend=%s", self.llm_model, type(active).__module__)
        tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
        conn_options = self.session.conn_options.llm_conn_options

        async def _run(backend: llm.LLM) -> AsyncGenerator[llm.ChatChunk, None]:
            async with backend.chat(
                chat_ctx=chat_ctx, tools=tools, tool_choice=tool_choice, conn_options=conn_options
            ) as stream:
                async for chunk in stream:
                    yield chunk

        started = False
        try:
            async for chunk in _run(active):
                started = True
                yield chunk
        except APIError as e:
            # A bad/unsupported model on the Inference gateway 404s here. Don't kill the
            # session — fall back to the Gemma default if nothing has streamed yet.
            if active is not self.session.llm and not started:
                logger.warning("LLM %r failed (%s); falling back to Gemma", self.llm_model, e)
                async for chunk in _run(self.session.llm):
                    yield chunk
            else:
                raise


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT(model=DEFAULT_STT),
        llm=openai.LLM(
            model=GEMMA_MODEL,
            base_url=GEMMA_BASE_URL,
            api_key=GEMMA_API_KEY,
        ),
        tts=inference.TTS(
            model=DEFAULT_TTS,
            voice="Sarah",
            extra_kwargs={"delivery_mode": "CREATIVE"},
            # Batch sentences up to 900 chars per request — as large as we can go
            # while staying under Inworld's 1000-char send_text limit (the server
            # auto-flushes past 1000). All chunks share one session/context, so
            # prosody stays continuous across the turn.
            tokenizer=tokenize.blingfire.SentenceTokenizer(max_token_len=900),
        ),
        vad=silero.VAD.load(),
        expressiveness=CONVERSATIONAL_EXPRESSIVENESS_PRESET,
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
        if model == agent.llm_model:
            return ""
        agent.llm_model = model
        if model != GEMMA_MODEL:
            agent._inference_llm.update_options(model=model)
        logger.info("switching LLM → %s", model)
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
            "llm": agent.llm_model,
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
