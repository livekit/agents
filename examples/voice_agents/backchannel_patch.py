import logging
import re
import json
import os
from livekit.agents.tokenize.basic import split_words
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo

logger = logging.getLogger("backchannel-patch")

# =============================================================================
# BACKCHANNEL WORD CONFIG
# =============================================================================
CONFIG_FILE = "filter_config.json"
BACKCHANNEL_WORDS = set()
COMMAND_WORDS = set()

def load_config():
    global BACKCHANNEL_WORDS, COMMAND_WORDS
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                BACKCHANNEL_WORDS = set(data.get("backchannel_words", []))
                COMMAND_WORDS = set(data.get("command_words", []))
            logger.info(f"Loaded {len(BACKCHANNEL_WORDS)} ignored words from {CONFIG_FILE}")
        else:
            logger.warning(f"{CONFIG_FILE} not found! Using empty defaults.")
            # Fallback defaults if verification fails
            BACKCHANNEL_WORDS = {"yeah", "ok", "mhmm", "okay"}
            COMMAND_WORDS = {"stop", "wait", "no"}
    except Exception as e:
        logger.error(f"Failed to load {CONFIG_FILE}: {e}")

# Load immediately on import
load_config()

# Global tracking for delta calculation
_last_processed_transcript = ""

def _extract_words(text: str) -> list:
    """Extract words from text, removing punctuation."""
    if not text:
        return []
    normalized = text.lower().strip()
    return [w for w in re.sub(r'[^\w\s-]', ' ', normalized).split() if w]

def _get_transcript_delta(full_transcript: str) -> str:
    """Get new words since last processed turn."""
    global _last_processed_transcript
    
    if not full_transcript:
        return ""
    
    full_words = _extract_words(full_transcript)
    last_words = _extract_words(_last_processed_transcript)
    
    if not full_words:
        return ""
    
    if not last_words:
        # No previous transcript, take last 3 words as safe bet for backchannel
        return " ".join(full_words[-3:])
    
    last_len = len(last_words)
    full_len = len(full_words)
    
    if full_len <= last_len:
        # Transcript didn't grow, take tail
        return " ".join(full_words[-3:])
    
    # Extract delta
    delta_words = full_words[last_len:]
    return " ".join(delta_words)

def _update_last_transcript(transcript: str) -> None:
    global _last_processed_transcript
    _last_processed_transcript = transcript

def _is_backchannel_only(text: str) -> bool:
    """Check if text contains ONLY backchannel words."""
    if not text or not text.strip():
        return True
    
    words = _extract_words(text)
    if not words: 
        return True
        
    # check commands first
    for word in words:
        if word in COMMAND_WORDS:
            return False
            
    # check all words
    for word in words:
        if word in BACKCHANNEL_WORDS:
            continue
        if '-' in word:
            parts = word.split('-')
            if all(p in BACKCHANNEL_WORDS for p in parts if p):
                continue
        return False
        
    return True

def _check_backchannel_delta(full_transcript: str) -> bool:
    delta = _get_transcript_delta(full_transcript)
    if not delta:
        return True # Empty delta is safe
        
    is_bc = _is_backchannel_only(delta)
    if is_bc:
        logger.info(f"🛡️ [PATCH FILTER] Delta '{delta}' is backchannel - BLOCKING")
    else:
        logger.debug(f"✅ [PATCH FILTER] Delta '{delta}' is NOT backchannel - allow")
    return is_bc

# =============================================================================
# PATCH METHODS
# =============================================================================

# Store original methods to call them if needed, or we just reimplement logic
# Reimplementing logic is safer to inject checks in middle

def patched_interrupt_by_audio_activity(self) -> None:
    # Copied logic from AgentActivity._interrupt_by_audio_activity
    # with injected checks
    opt = self._session.options
    use_pause = opt.resume_false_interruption and opt.false_interruption_timeout is not None

    if hasattr(self.llm, "capabilities") and self.llm.capabilities.turn_detection:
        return

    if (
        self.stt is not None
        and opt.min_interruption_words > 0
        and self._audio_recognition is not None
    ):
        text = self._audio_recognition.current_transcript
        
        # Original count check
        if len(split_words(text, split_character=True)) < opt.min_interruption_words:
            return
            
        # --- PATCH START ---
        if _check_backchannel_delta(text):
            logger.debug(f"🚫 [PATCH FILTER] Blocking AUDIO interruption - backchannel delta")
            return
        # --- PATCH END ---

    if self._rt_session is not None:
        self._rt_session.start_user_activity()

    if (
        self._current_speech is not None
        and not self._current_speech.interrupted
        and self._current_speech.allow_interruptions
    ):
        self._paused_speech = self._current_speech

        if self._false_interruption_timer:
            self._false_interruption_timer.cancel()
            self._false_interruption_timer = None

        if use_pause and self._session.output.audio and self._session.output.audio.can_pause:
            # THIS IS THE CRITICAL LINE THAT WAS PAUSING
            self._session.output.audio.pause()
            self._session._update_agent_state("listening")
        else:
            if self._rt_session is not None:
                self._rt_session.interrupt()
            self._current_speech.interrupt()


def patched_on_end_of_turn(self, info: _EndOfTurnInfo) -> bool:
    # Reimplement logic with checks
    if self._scheduling_paused:
        self._cancel_preemptive_generation()
        # ... logs ...
        if self._session._closing:
             # handle closing... simplified for patch
             pass
        return True

    text_len = len(split_words(info.new_transcript, split_character=True))
    if (
        self.stt is not None
        and self._turn_detection != "manual"
        and self._current_speech is not None
        and self._current_speech.allow_interruptions
        and not self._current_speech.interrupted
        and self._session.options.min_interruption_words > 0
        and text_len < self._session.options.min_interruption_words
    ):
        self._cancel_preemptive_generation()
        return False
        
    # --- PATCH START ---
    if (
        self._current_speech is not None 
        and not self._current_speech.interrupted
        and _check_backchannel_delta(info.new_transcript)
    ):
        self._cancel_preemptive_generation()
        logger.info(f"🚫 [PATCH FILTER] EndOfTurn blocked - backchannel delta")
        return False
        
    _update_last_transcript(info.new_transcript)
    # --- PATCH END ---

    old_task = self._user_turn_completed_atask
    self._user_turn_completed_atask = self._create_speech_task(
        self._user_turn_completed_task(old_task, info),
        name="AgentActivity._user_turn_completed_task",
    )
    return True


def apply_patch():
    logger.info("Applying Backchannel Monkeypatch to AgentActivity...")
    AgentActivity._interrupt_by_audio_activity = patched_interrupt_by_audio_activity
    AgentActivity.on_end_of_turn = patched_on_end_of_turn
    logger.info("✅ Monkeypatch applied successfully!")
