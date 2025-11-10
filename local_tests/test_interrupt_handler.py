from livekit.extension.interrupt_handler import InterruptHandler, ASRChunk, InterruptDecision
from livekit.extension.config import InterruptConfig
import pytest

def _mk(cfg: InterruptConfig | None = None) -> InterruptHandler:
    return InterruptHandler(cfg)


def test_ignore_fillers_while_speaking():
    cfg = InterruptConfig(
        ignored_words={"um", "uh", "hmm"},
        priority_words={"stop"},
        asr_conf_min=0.4,
        filler_ratio_min=0.7,
    )
    ih = _mk(cfg)
    ih.on_tts_start()  # agent is speaking
    decision, meta = ih.classify(ASRChunk("um uh hmm", 0.9, False))
    assert decision == InterruptDecision.IGNORE
    assert meta["reason"] == "filler_dominated_while_speaking"
    assert meta["agent_speaking"] is True
    assert meta["filler_ratio"] >= 0.7


def test_accept_when_quiet_even_if_filler():
    ih = _mk()
    ih.on_tts_end()  # agent is quiet
    decision, meta = ih.classify(ASRChunk("umm", 0.2, True))
    assert decision == InterruptDecision.ACCEPT
    assert meta["reason"] == "agent_quiet_accept"
    assert meta["agent_speaking"] is False


def test_force_stop_on_priority_phrase():
    ih = _mk()
    ih.on_tts_start()  # speaking
    decision, meta = ih.classify(ASRChunk("wait one second", 0.3, False))
    assert decision == InterruptDecision.FORCE_STOP
    assert meta["reason"] == "priority_word"


def test_low_confidence_is_ignored_while_speaking():
    ih = _mk()
    ih.on_tts_start()
    decision, meta = ih.classify(ASRChunk("hmm yeah", 0.3, False))
    assert decision == InterruptDecision.IGNORE
    assert meta["reason"] == "low_conf_suppress_while_speaking"


def test_accept_real_interruption_while_speaking():
    cfg = InterruptConfig(
        ignored_words={"um", "uh"},
        priority_words={"stop"},
        asr_conf_min=0.5,
        filler_ratio_min=0.8,
    )
    ih = _mk(cfg)
    ih.on_tts_start()
    # Mixed content with few/no fillers and decent confidence
    decision, meta = ih.classify(ASRChunk("no please continue later", 0.92, True))
    # Could be ACCEPT (real speech) unless "no" is in priority_words
    # Our cfg priority_words only has "stop", so expect ACCEPT
    assert decision == InterruptDecision.ACCEPT
    assert meta["reason"] == "real_interruption_while_speaking"


