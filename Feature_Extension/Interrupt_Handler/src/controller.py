import time
from dataclasses import dataclass
from typing import Optional, Callable, List, Any, Awaitable
from .config import IHConfig
from .classifier import UtteranceClassifier
from .state import SpeechGate
from .logkit import make_logger

log = make_logger("interrupt-orchestrator")

@dataclass
class ASRChunk:
    text: str
    is_final: bool = False
    confidence: Optional[float] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None

    @property
    def duration_ms(self) -> Optional[int]:
        if self.start_ms is None or self.end_ms is None:
            return None
        return max(0, self.end_ms - self.start_ms)

class MicroBuffer:
    """Tiny debounce buffer to merge rapid partials without adding lag."""
    def __init__(self, window_ms: int):
        self.window_ms = window_ms
        self._buf: List[ASRChunk] = []
        self._last_add: float = 0.0

    def add(self, e: ASRChunk):
        now = time.time()
        if not self._buf or (now - self._last_add) * 1000 <= self.window_ms:
            self._buf.append(e)
        else:
            self._buf = [e]
        self._last_add = now

    def merged(self) -> ASRChunk:
        if not self._buf:
            return ASRChunk(text="", is_final=False)
        txt = " ".join(x.text for x in self._buf if x.text)
        finals = any(x.is_final for x in self._buf)
        confs = [x.confidence for x in self._buf if x.confidence is not None]
        conf = sum(confs) / len(confs) if confs else None
        s = min([x.start_ms for x in self._buf if x.start_ms is not None], default=None)
        e = max([x.end_ms for x in self._buf if x.end_ms is not None], default=None)
        return ASRChunk(text=txt, is_final=finals, confidence=conf, start_ms=s, end_ms=e)

    def clear(self):
        self._buf.clear()

class InterruptOrchestrator:
    """
    Wires LiveKit session events to semantic decisions:
      - When agent is speaking: ignore fillers & low conf; interrupt on HARD_INTENT or CONTENT.
      - When agent is quiet: forward everything (even fillers).
    """
    def __init__(
        self,
        session: Any,                   # LiveKit AgentSession (duck-typed)
        state: SpeechGate,
        config: IHConfig,
        forward_user_text: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.session = session
        self.state = state
        self.cfg = config
        self.classifier = UtteranceClassifier(config)
        self.buf = MicroBuffer(config.debounce_ms)
        self.forward_user_text = forward_user_text

    # Hook these to LiveKit signals:
    def on_tts_started(self, *_):
        self.state.open()
        log.debug("TTS started")

    def on_tts_finished(self, *_):
        self.state.close()
        log.debug("TTS ended")

    async def on_transcription(self, text: str, confidence: Optional[float] = None,
                               is_final: bool = False, start_ms: Optional[int] = None,
                               end_ms: Optional[int] = None):
        chunk = ASRChunk(text=text, confidence=confidence, is_final=is_final,
                         start_ms=start_ms, end_ms=end_ms)
        self.buf.add(chunk)
        merged = self.buf.merged()
        verdict = self.classifier.decide(merged.text, merged.confidence, merged.duration_ms)

        if not self.state.speaking:
            # Agent is quiet → treat as normal user speech
            if self.forward_user_text:
                await self.forward_user_text(merged.text)
            log.info(f"PASS_THROUGH | {verdict.label} | {merged.text}")
            return "PASS"

        # Agent is speaking → decide interrupt vs ignore
        if verdict.label in ("HARD_INTENT", "CONTENT"):
            self.buf.clear()
            if hasattr(self.session, "interrupt"):
                await self.session.interrupt()  # standard AgentSession API
            log.info(f"INTERRUPT | {verdict.label}:{verdict.reason} | {merged.text}")
            return "INTERRUPT"

        # Otherwise ignored while speaking
        log.debug(f"IGNORED | {verdict.label}:{verdict.reason} | {merged.text}")
        return "IGNORE"
