import logging
import re

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
    AgentStateChangedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()

# ----------------- WORD LISTS (from config) -----------------

from interrupt_config import SOFT_WORDS, HARD_WORDS

def _tokens(text: str) -> list[str]:
    # split on non-letters so "ok," / "yeah?" etc still work
    return [w for w in re.split(r"[^a-z]+", text.lower()) if w]


def _is_soft(words: list[str]) -> bool:
    return bool(words) and all(w in SOFT_WORDS for w in words)


def _has_hard(words: list[str]) -> bool:
    return any(w in HARD_WORDS for w in words)


# ----------------- AGENT -----------------


class MyAgent(Agent):
    def __init__(self) -> None:
        # we do NOT set allow_interruptions here; we control it at session-level
        super().__init__(
            instructions=(
                "Your name is Kelly. You interact with users via voice. "
                "Keep responses concise and to the point. "
                "Do not use emojis, asterisks, markdown, or other special characters. "
                "You are curious and friendly, and have a sense of humor. "
                "You will speak English to the user."
            ),
        )

    async def on_enter(self):
        logger.info("[AGENT] on_enter -> generating initial reply")
        self.session.generate_reply()

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        """Called when the user asks for weather related information."""
        logger.info(f"[TOOL] Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    logger.info("[PREWARM] loading VAD model")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("[PREWARM] VAD model loaded")


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    logger.info("[ENTRYPOINT] starting AgentSession for room=%s", ctx.room.name)

    session = AgentSession(
        # REQUIRED MODELS
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-nano",
        tts="inworld/inworld-tts-1",

        # Turn detection + VAD
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],

        preemptive_generation=True,

        # false interruption handling (still useful)
        resume_false_interruption=True,
        false_interruption_timeout=1.0,

        # 🔑 KEY SETTINGS:
        # DO NOT allow automatic interruptions of TTS by user speech.
        allow_interruptions=False,
        # BUT still keep user audio and send to STT while uninterruptible.
        discard_audio_if_uninterruptible=False,
    )

    # ------------- METRICS -------------

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"[METRICS] Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # ------------- DEBUG STATE LOGS -------------

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        logger.info(
            "[STATE] agent_state changed: %s -> %s | user_state=%s",
            ev.old_state,
            ev.new_state,
            session.user_state,
        )

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: UserStateChangedEvent):
        logger.info(
            "[STATE] user_state changed: %s -> %s | agent_state=%s",
            ev.old_state,
            ev.new_state,
            session.agent_state,
        )

    # ------------- CORE STT LOGIC -------------

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
        text = (ev.transcript or "").strip()
        agent_state = session.agent_state

        logger.info(
            "[STT] %s text=%r | agent_state=%s | user_state=%s",
            "FINAL" if ev.is_final else "PARTIAL",
            text,
            agent_state,
            session.user_state,
        )

        if not text:
            return

        # Only act on FINAL transcripts (partials are too noisy)
        if not ev.is_final:
            return

        words = _tokens(text)
        is_soft = _is_soft(words)
        has_hard = _has_hard(words)

        logger.info(
            "[LOGIC] words=%s | is_soft=%s | has_hard=%s | agent_state=%s",
            words,
            is_soft,
            has_hard,
            agent_state,
        )

        # ----- CASE 1: Agent is SPEAKING -----
        if agent_state == "speaking":
            # HARD interrupt: "stop / wait / no / cancel / pause"
            if has_hard:
                logger.info(
                    "[LOGIC] HARD interrupt while speaking -> calling session.interrupt(force=True)"
                )
                try:
                    # 🔥 IMPORTANT: force=True so we can override non-interruptible handles
                    session.interrupt(force=True)
                except Exception:
                    logger.exception(
                        "[ERROR] session.interrupt(force=True) raised an exception"
                    )
                return

            # PURE SOFT backchannel: only yeah/ok/hmm, etc.
            if is_soft:
                logger.info(
                    "[LOGIC] SOFT backchannel while speaking -> IGNORE COMPLETELY "
                    "(no interrupt, no new turn)"
                )
                # do nothing: agent continues talking, no hiccup, no new LLM turn
                return

            # Other content while speaking: treat as real interrupt for now
            logger.info(
                "[LOGIC] NON-SOFT utterance while speaking -> treating as interrupt "
                "(session.interrupt(force=True))"
            )
            try:
                session.interrupt(force=True)
            except Exception:
                logger.exception(
                    "[ERROR] session.interrupt(force=True) raised an exception"
                )
            return

        # ----- CASE 2: Agent is NOT SPEAKING -----
        # In this case, we let everything pass through as normal user input.
        # This gives you: "yeah/ok/hmm" while silent -> agent responds.
        if agent_state != "speaking":
            if is_soft:
                logger.info(
                    "[LOGIC] SOFT input while agent NOT speaking -> normal user turn "
                    "(agent will respond)"
                )
            elif has_hard:
                logger.info(
                    "[LOGIC] HARD word while agent NOT speaking -> normal user turn "
                    "(LLM decides what to do)"
                )
            else:
                logger.info(
                    "[LOGIC] NORMAL input while agent NOT speaking -> normal user turn"
                )
            # No special actions; just let the framework handle it.
            return

    # ------------- START SESSION -------------

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s %(levelname)-5s %(name)-12s %(message)s",
#     )
#     cli.run_app(server)

if __name__ == "__main__":
    import os

    # Make sure proof/ exists
    os.makedirs("proof", exist_ok=True)

    # Console logging (optional, for debugging)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)-12s %(message)s",
    )

    # File logging for assignment proof
    file_handler = logging.FileHandler(
        "proof/log-transcript-harshmehta1618.txt",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-5s %(name)-12s %(message)s")
    )

    # Attach to root logger so *all* logs (your [LOGIC], [STT], livekit.agents, etc.) go there
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    cli.run_app(server)
