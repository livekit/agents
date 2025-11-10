import logging
from .interrupt_handler import InterruptHandler, ASRChunk, InterruptDecision

log = logging.getLogger("middleware")
log.setLevel(logging.INFO)

class LiveKitInterruptMiddleware:
    def __init__(self, agent, cfg=None):
        """
        agent: your LiveKit agent instance, which exposes:
          - events/callbacks for TTS start/stop
          - ASR transcription events (partial/final with confidence)
          - methods: agent.stop_tts(), agent.enqueue_user_utterance(text), etc.
        """
        self.agent = agent
        self.ih = InterruptHandler(cfg)

        # Hook up to agent events (pseudo-API; adapt to your SDK)
        agent.on("tts_start", self.ih.on_tts_start)
        agent.on("tts_end", self.ih.on_tts_end)
        agent.on("transcript", self._on_transcript)

    def _on_transcript(self, text: str, confidence: float, is_final: bool, raw=None):
        chunk = ASRChunk(text=text, avg_confidence=confidence, is_final=is_final)
        decision, meta = self.ih.classify(chunk)

        if decision == InterruptDecision.FORCE_STOP:
            log.info("[INTERRUPT] FORCE_STOP %s", meta)
            self.agent.stop_tts()
            # You may also route the utterance downstream if desired:
            self.agent.enqueue_user_utterance(text)
            return

        if decision == InterruptDecision.ACCEPT:
            log.info("[INTERRUPT] ACCEPT %s", meta)
            if self.ih.agent_speaking:
                # we stop speaking to respect the user
                self.agent.stop_tts()
            self.agent.enqueue_user_utterance(text)
            return

        # IGNORE
        log.info("[INTERRUPT] IGNORE %s", meta)
        # Do nothing (keep speaking)
