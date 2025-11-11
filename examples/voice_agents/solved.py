# examples/voice_agents/basic_agent.py
import asyncio
import logging
import os
import re
import sys
from typing import Iterable, Optional, Set, Any
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from dotenv import load_dotenv

# --- livekit imports (unchanged) ---
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# --- logging + dotenv ---
logger = logging.getLogger("basic-agent")
load_dotenv()


logger_f = logging.getLogger("livekit.filler_filter")
if not logger_f.handlers:
    logger_f.setLevel(logging.INFO)

DEFAULT_IGNORED = {"uh", "umm", "hmm", "haan", "mm", "mhm", "uhh", "uhm", "achha", "arre"}
WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)


def _normalize_token(token: str) -> str:
    t = token.lower().strip()
    t = re.sub(r"^[^\w']+|[^\w']+$", "", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return t


def _get_field(event: Any, name: str, default=None):
    try:
        if isinstance(event, dict):
            return event.get(name, default)
        return getattr(event, name, default)
    except Exception:
        return default


class FillerInterruptFilter:
    def __init__(
        self,
        session,
        ignored_words: Optional[Iterable[str]] = None,
        min_confidence: float = 0.45,
        enable_confidence_filter: bool = True,
        filler_ratio_threshold: float = 0.8,
        min_non_filler_for_low_confidence: int = 2,
        attach_events: Iterable[str] = ("user_input_transcribed", "transcription"),
    ):
        self.session = session
        self.ignored_words: Set[str] = set(DEFAULT_IGNORED)
        if ignored_words:
            self.ignored_words.update(w.lower() for w in ignored_words)
        self.min_confidence = float(min_confidence)
        self.enable_confidence_filter = bool(enable_confidence_filter)
        self.filler_ratio_threshold = float(filler_ratio_threshold)
        self.min_non_filler_for_low_confidence = int(min_non_filler_for_low_confidence)
        self._lock = asyncio.Lock()

        logger_f.info(
            "FillerInterruptFilter created; ignored_words=%s min_confidence=%s filler_ratio=%s",
            sorted(self.ignored_words),
            self.min_confidence,
            self.filler_ratio_threshold,
        )

        attached = False
        for ev_name in attach_events:
            try:
                def _make_wrapper(cb):
                    def _wrapper(evt):
                        try:
                            asyncio.create_task(cb(evt))
                        except Exception as ex:
                            logger_f.exception("Error scheduling filler filter callback: %s", ex)
                    return _wrapper

                try:
                    session.on(ev_name, _make_wrapper(self._on_transcribed))
                    attached = True
                    logger_f.debug("FillerInterruptFilter attached to event '%s' via session.on", ev_name)
                    break
                except TypeError:
                    try:
                        decorator = session.on(ev_name)
                        decorator(_make_wrapper(self._on_transcribed))
                        attached = True
                        logger_f.debug("FillerInterruptFilter attached to event '%s' via session.on decorator", ev_name)
                        break
                    except Exception:
                        pass
            except Exception:
                pass

        if not attached:
            logger_f.warning(
                "Could not attach filler filter to session events using names %s. Check AgentSession's event API.",
                attach_events,
            )

    def update_ignored_words(self, new_list: Iterable[str], replace: bool = False):
        if replace:
            self.ignored_words = set(w.lower() for w in new_list)
        else:
            self.ignored_words.update(w.lower() for w in new_list)
        logger_f.info("Updated filler ignored words: %s", sorted(self.ignored_words))

    async def _on_transcribed(self, event: Any):
        async with self._lock:
            transcript = _get_field(event, "transcript", None)
            if not transcript:
                return
            is_final = _get_field(event, "is_final", None)
            transcript_str = str(transcript).strip()
            if not transcript_str:
                return
            raw_tokens = WORD_RE.findall(transcript_str)
            tokens = [_normalize_token(t) for t in raw_tokens if _normalize_token(t)]
            if not tokens:
                if await self._is_agent_speaking():
                    await self._log_ignored(transcript_str, reason="empty_tokens")
                    await self._attempt_resume_tts()
                else:
                    await self._log_valid(transcript_str, None)
                return
            confidence = _get_field(event, "confidence", None)
            try:
                if confidence is not None:
                    confidence = float(confidence)
            except Exception:
                confidence = None
            agent_speaking = await self._is_agent_speaking()
            filler_tokens = [t for t in tokens if t in self.ignored_words]
            non_filler_tokens = [t for t in tokens if t not in self.ignored_words]
            filler_ratio = len(filler_tokens) / float(max(1, len(tokens)))
            effective_ratio_threshold = self.filler_ratio_threshold if is_final is not False else min(0.95, self.filler_ratio_threshold + 0.15)

            if agent_speaking:
                if filler_ratio >= effective_ratio_threshold:
                    await self._log_ignored(transcript_str, reason=f"filler_ratio({filler_ratio:.2f})")
                    await self._attempt_resume_tts()
                    return
                if self.enable_confidence_filter and (confidence is not None):
                    if (confidence < self.min_confidence) and (len(non_filler_tokens) < self.min_non_filler_for_low_confidence):
                        await self._log_ignored(transcript_str, reason=f"low_confidence({confidence:.2f})")
                        await self._attempt_resume_tts()
                        return
                await self._log_valid(transcript_str, confidence)
                return
            else:
                await self._log_valid(transcript_str, confidence)
                return

    async def _is_agent_speaking(self) -> bool:
        s = self.session
        try:
            if hasattr(s, "is_speaking"):
                val = getattr(s, "is_speaking")
                if callable(val):
                    try:
                        res = val()
                        if asyncio.iscoroutine(res):
                            res = await res
                    except Exception:
                        res = False
                    return bool(res)
                return bool(val)
        except Exception:
            pass
        try:
            tts = getattr(s, "tts", None)
            if tts is not None:
                if hasattr(tts, "is_speaking"):
                    fn = getattr(tts, "is_speaking")
                    res = fn() if callable(fn) else fn
                    if asyncio.iscoroutine(res):
                        res = await res
                    return bool(res)
                if hasattr(tts, "speaking"):
                    sp = getattr(tts, "speaking")
                    res = sp() if callable(sp) else sp
                    if asyncio.iscoroutine(res):
                        res = await res
                    return bool(res)
        except Exception:
            pass
        try:
            if hasattr(s, "_current_speech"):
                return getattr(s, "_current_speech") is not None
            if hasattr(s, "current_speech"):
                return getattr(s, "current_speech") is not None
        except Exception:
            pass
        return False

    async def _attempt_resume_tts(self):
        try:
            tts = getattr(self.session, "tts", None)
            if not tts:
                return
            for name in ("resume", "unpause", "play", "continue_playback"):
                if hasattr(tts, name):
                    fn = getattr(tts, name)
                    try:
                        maybe = fn() if callable(fn) else fn
                        if asyncio.iscoroutine(maybe):
                            await maybe
                        logger_f.debug("[filler_filter] called tts.%s to resume", name)
                        return
                    except Exception:
                        logger_f.debug("[filler_filter] calling tts.%s failed", name, exc_info=True)
                        continue
            try:
                self.session.emit("filler_resume_suggested", {"source": "filler_interrupt_filter"})
            except Exception:
                pass
        except Exception:
            logger_f.exception("Error attempting to resume TTS")

    async def _log_ignored(self, transcript: str, reason: str = "filler"):
        logger_f.info("[filler_filter|ignored] %s (reason=%s)", transcript, reason)
        try:
            self.session.emit("filler_ignored", {"transcript": transcript, "reason": reason})
        except Exception:
            pass

    async def _log_valid(self, transcript: str, confidence: Optional[float] = None):
        logger_f.info("[filler_filter|valid] %s (confidence=%s)", transcript, confidence)
        try:
            self.session.emit("filler_valid", {"transcript": transcript, "confidence": confidence})
        except Exception:
            pass

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "do not use emojis, asterisks, markdown, or other special characters in your responses."
            "You are curious and friendly, and have a sense of humor."
            "you will speak english to the user",
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        await self.session.generate_reply()

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."



def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["turn_detector"] = MultilingualModel.load()


async def entrypoint(ctx: JobContext):

    ctx.log_context_fields = {"room": ctx.room.name}
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",

        turn_detection=ctx.proc.userdata["turn_detector"], 

        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,

        resume_false_interruption=True, 

        false_interruption_timeout=1.0,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    # Attach filler filter (now embedded)
    filter_plugin = FillerInterruptFilter(
        session,
        ignored_words=["uh", "umm", "hmm", "haan"],
        min_confidence=0.45,
    )
    filter_plugin.update_ignored_words(["achha", "arre"], replace=False)


# ---------------------------------------------------------
# Local quick plugin test (run without LiveKit by setting env variable)
# ---------------------------------------------------------
def _run_local_plugin_test_and_exit():
    import json

    class DummySession:
        def __init__(self, speaking=False):
            self._speaking = speaking
            self._events = []

        def on(self, evname, cb=None):
            if cb is None:
                def decorator(fn):
                    self._events.append((evname, fn))
                    return fn
                return decorator
            else:
                self._events.append((evname, cb))
                return None

        def emit(self, name, payload=None):
            self._events.append(("emit:" + name, payload))

        @property
        def is_speaking(self):
            return self._speaking

    async def _run_test():
        print(">>> Running local filler-interrupt-filter test (no LiveKit required)")
        s = DummySession(speaking=True)
        f = FillerInterruptFilter(s, ignored_words=["uh", "umm", "haan", "achha"], min_confidence=0.45)
        tests = [
            {"transcript": "uh umm", "confidence": 0.9, "is_final": True},
            {"transcript": "uh hello", "confidence": 0.9, "is_final": True},
            {"transcript": "hmm", "confidence": 0.2, "is_final": True},
            {"transcript": "uhh uh", "confidence": 0.5, "is_final": False},
            {"transcript": "achha", "confidence": 0.9, "is_final": True},
            {"transcript": "I want pizza", "confidence": 0.95, "is_final": True},
        ]
        for ev in tests:
            await f._on_transcribed(ev)
        print("\nCaptured emits & events (DummySession):")
        for e in s._events:
            print(e)
        print(">>> Local test finished.")

    asyncio.run(_run_test())
    sys.exit(0)

if os.getenv("LOCAL_PLUGIN_TEST", "0") == "1":
    _run_local_plugin_test_and_exit()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
