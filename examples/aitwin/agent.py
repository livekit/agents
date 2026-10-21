"""
LiveKit Agents worker — STT + LLM + TTS; AiTwin server aligns visemes.

Pipeline:
  user mic / lk.chat → STT → OpenAI LLM → text
  → OpenAI TTS PCM (or HeadTTS)
  → AiTwin POST /avatar_alignVisemes (server G2P / alignment)
  → publish topic aitwin.speech (begin / chunk / end JSON)
  → browser AiTwin beginSpeech / pushSpeechChunk / endSpeech

Agent does not publish a LiveKit audio track (audio_output=False) so playback
stays clocked with visemes inside AiTwin Web Audio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    ConversationItemAddedEvent,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice.room_io import RoomOptions
from livekit.plugins import openai, silero

from speech_synth import synthesize_speech

load_dotenv()

logger = logging.getLogger("aitwin-livekit-demo")
logger.setLevel(logging.INFO)


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name} (set it in examples/aitwin/.env)")
    return value


AGENT_NAME = _require_env("AGENT_NAME")
_LLM_MODEL = _require_env("LLM_MODEL")
_STT_MODEL = _require_env("STT_MODEL")
TOPIC_SPEECH = _require_env("TOPIC_SPEECH")
# Exact words from the UI "Speak text" mode (no LLM) → same TTS+align path.
TOPIC_SAY = (os.getenv("TOPIC_SAY") or "aitwin.say").strip()
# Viseme profile engine id understood by @streamoji/aitwin (Cartesia = oculus_direct).
AITWIN_VISEME_ENGINE_ID = (os.getenv("AITWIN_VISEME_ENGINE_ID") or "eng_c9b1e6d4").strip()


class VoiceAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful assistant whose replies are spoken by an AiTwin avatar. "
                "Keep replies to 1-2 short sentences. Prefer fast, natural speech."
            )
        )


async def entrypoint(ctx: JobContext) -> None:
    logger.info("Connecting to room: %s", ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(model=_STT_MODEL),
        llm=openai.LLM(model=_LLM_MODEL),
        # TTS runs in-process via speech_synth; do not use AgentSession TTS publish.
        tts=None,
    )

    speak_lock = asyncio.Lock()

    async def publish_json(payload: dict) -> None:
        if not ctx.room.isconnected():
            return
        body = json.dumps(payload, separators=(",", ":"))
        await ctx.room.local_participant.send_text(body, topic=TOPIC_SPEECH)

    async def publish_speech(text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        async with speak_lock:
            try:
                bundle = await synthesize_speech(cleaned)
            except Exception:
                logger.exception("speech synth failed for: %s", cleaned[:120])
                return

            try:
                await publish_json(
                    {
                        "type": "begin",
                        "text": bundle.text,
                        "ttsEngineId": AITWIN_VISEME_ENGINE_ID,
                        "sampleRate": bundle.sample_rate,
                    }
                )
                await publish_json(
                    {
                        "type": "chunk",
                        "audio": bundle.audio_b64,
                        "sampleRate": bundle.sample_rate,
                        "visemes": bundle.visemes,
                        "words": bundle.words,
                        "isNewSegment": True,
                        "audioOffsetMs": 0,
                    }
                )
                await publish_json({"type": "end"})
                logger.info(
                    "published %s (%d chars, %d visemes, sr=%d)",
                    TOPIC_SPEECH,
                    len(bundle.text),
                    len(bundle.visemes),
                    bundle.sample_rate,
                )
            except Exception:
                logger.warning("failed to publish %s", TOPIC_SPEECH, exc_info=True)
                try:
                    await publish_json({"type": "end"})
                except Exception:
                    pass

    @session.on("conversation_item_added")
    def _on_conversation_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if not isinstance(item, llm.ChatMessage) or item.role != "assistant":
            return
        text = (item.text_content or item.raw_text_content or "").strip()
        if not text:
            return
        asyncio.create_task(publish_speech(text))

    # Speak-text UI: exact words, no LLM — same OpenAI TTS + /avatar_alignVisemes path.
    # LiveKit TextStreamHandler is Callable[[TextStreamReader, str], None] (identity str).
    def _on_say_stream(reader, participant_identity: str) -> None:
        async def _read_and_speak() -> None:
            try:
                text = (await reader.read_all()).strip()
            except Exception:
                logger.warning("%s stream read failed", TOPIC_SAY, exc_info=True)
                return
            if not text:
                return
            if participant_identity == ctx.room.local_participant.identity:
                return
            logger.info("say request from %s: %s", participant_identity, text[:120])
            await publish_speech(text)

        asyncio.create_task(_read_and_speak())

    ctx.room.register_text_stream_handler(TOPIC_SAY, _on_say_stream)

    await session.start(
        agent=VoiceAssistant(),
        room=ctx.room,
        room_options=RoomOptions(
            audio_input=True,
            text_input=True,
            audio_output=False,
            text_output=False,
            video_input=False,
        ),
    )

    await session.generate_reply(
        instructions="Greet the user in one short friendly sentence.",
    )


def main() -> None:
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=AGENT_NAME,
            num_idle_processes=1,
        )
    )


if __name__ == "__main__":
    main()
