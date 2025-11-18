# extensions/interrupt_handler/handler.py
import re
from typing import List
from .config import IGNORED_WORDS, ASR_CONFIDENCE_THRESHOLD, COMMAND_KEYWORDS
from .logger import log

_token_re = re.compile(r"\w+")

def tokenize(text: str) -> List[str]:
    return _token_re.findall((text or "").lower())

def is_filler_only(text: str, confidence: float) -> bool:
    tokens = tokenize(text)
    if not tokens:
        return True
    
    if any(t in COMMAND_KEYWORDS for t in tokens):
        return False
    
    non_filler = [t for t in tokens if t not in IGNORED_WORDS]
    if non_filler:
        return False
    
    return confidence >= ASR_CONFIDENCE_THRESHOLD

async def decide_and_handle(transcript: str, confidence: float, agent_speaking: bool, on_interrupt_cb):
    """
    Decide whether to ignore or treat as interrupt.
    on_interrupt_cb is an async callable to call when we must interrupt.
    """
    tokens = tokenize(transcript)
    filler = is_filler_only(transcript, confidence)
    if agent_speaking and filler:
        log("IGNORED_FILLER", transcript=transcript, tokens=tokens, confidence=confidence, agent_speaking=agent_speaking)
        return {"action": "ignored"}
    
    log("VALID_INTERRUPT", transcript=transcript, tokens=tokens, confidence=confidence, agent_speaking=agent_speaking)
  
    try:
        await on_interrupt_cb(transcript, confidence)
    except Exception as e:
        log("INTERRUPT_CB_ERROR", error=str(e))
    return {"action": "interrupt"}
