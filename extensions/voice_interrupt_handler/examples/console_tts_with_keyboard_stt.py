# console_tts_with_keyboard_stt_debug_force_stop.py
"""
Debug + Force-Stop demo.
When you type a stop-word (e.g. "stop"), this demo will IMMEDIATELY call mock.interrupt()
and stop the local TTS engine. This bypasses any internal handler heuristics so we can
observe whether stopping audio works in your environment.
"""

import os, sys, asyncio, threading, logging, time, functools

ext = os.path.join(os.getcwd(), "extensions", "voice_interrupt_handler")
if ext not in sys.path:
    sys.path.insert(0, ext)

from voice_interrupt.handler import InterruptHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("console_tts_debug_force")

# --- MockSession that stops TTS on interrupt ---
class MockSession:
    def __init__(self, tts_engine=None):
        self._callbacks = {}
        self.interrupted = 0
        self._tts_engine = tts_engine

    def on(self, name, cb):
        logger.info(f"[MockSession] registered callback -> {name}")
        self._callbacks[name] = cb

    def interrupt(self):
        self.interrupted += 1
        logger.info("[MockSession] session.interrupt() called")
        try:
            if self._tts_engine is not None:
                logger.info("[MockSession] attempting to stop TTS engine now")
                self._tts_engine.stop()
                logger.info("[MockSession] TTS engine.stop() returned")
        except Exception:
            logger.exception("Error stopping TTS engine on interrupt")

# --- pyttsx3 TTS helpers ---
def init_tts():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)
        return engine
    except Exception as e:
        logger.warning("pyttsx3 init failed: %s", e)
        return None

def speak(engine, text):
    if engine:
        def _s():
            try:
                logger.info(f"[TTS] speaking: {text!r}")
                engine.say(text)
                engine.runAndWait()
                logger.info("[TTS] runAndWait finished")
            except Exception:
                logger.exception("TTS error")
        threading.Thread(target=_s, daemon=True).start()
    else:
        logger.info("[TTS missing] would speak: %s", text)

# --- Main ---
async def main():
    engine = init_tts()
    mock = MockSession(tts_engine=engine)

    handler = InterruptHandler(
        mock,
        ignored_words={"uh", "umm", "hmm", "haan"},
        stop_words={"stop", "wait"},
        min_confidence=0.0
    )
    handler.start()
    logger.info("[Main] Handler started and attached to mock session")

    speak(engine, "Hello. This force-stop debug demo uses local TTS. Type lines to simulate transcripts. Type 'stop' to force an immediate stop.")

    loop = asyncio.get_running_loop()
    cancel_event = asyncio.Event()

    def stdin_reader():
        logger.info("Type lines to simulate ASR transcript (type 'exit' to quit). 'stop' will trigger immediate interrupt.")
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    continue
                line = line.strip()
                logger.info(f"[stdin] read line: {line!r}")

                if line.lower() == "exit":
                    logger.info("[stdin] exit received")
                    loop.call_soon_threadsafe(cancel_event.set)
                    break

                # Always schedule handler like before:
                evt = {"is_final": True, "transcript": line, "confidence": 1.0}
                logger.info(f"[stdin] scheduling handler._on_user_input_transcribed with evt={evt!r}")
                loop.call_soon_threadsafe(asyncio.create_task, handler._on_user_input_transcribed(evt))

                # IMMEDIATE FORCE: if the typed word is a stop-word, call interrupt() directly
                if line.lower() in {"stop", "wait"}:
                    logger.info("[stdin] Detected typed stop-word; calling mock.interrupt() IMMEDIATELY to force stop audio")
                    loop.call_soon_threadsafe(mock.interrupt)

            except Exception:
                logger.exception("stdin reader encountered exception")
                break

    threading.Thread(target=stdin_reader, daemon=True).start()

    await cancel_event.wait()
    logger.info("Force-stop debug demo stopped cleanly")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
