"""
LiveKit Voice Interruption Handling Agent

Description:
    This module implements a LiveKit Agent tailored for the "Voice Interruption Handling Challenge."
    It extends standard Voice Activity Detection (VAD) by adding a semantic filtering layer
    that distinguishes between meaningful user interruptions and non-disruptive filler words 
    (e.g., "um", "uh", "hmmm").

Key Features:
    - Real-time Audio Transcription: Uses Deepgram STT for fast ASR.
    - Dynamic Filler Configuration: Loads ignored words from an external file, allowing runtime updates.
    - Smart Normalization: Handles character elongation (e.g., "hmmmm" -> "hmm") and punctuation.
    - Command Recognition: Detects immediate stop commands to override filler logic.

Dependencies:
    - livekit-agents
    - livekit-plugins-deepgram
    - python-dotenv

Author: [Your Name/SalesCode Candidate]
Date: 2025-10-13
"""

import asyncio
import logging
import os
import re
import string
from dotenv import load_dotenv

# LiveKit Imports
from livekit.agents import cli, WorkerOptions, JobContext, stt
from livekit import rtc
from livekit.plugins import deepgram, silero

load_dotenv()

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))

# Path to the dynamic configuration file for ignored words
IGNORED_WORDS_FILE = os.path.join(REPO_ROOT, "ignored_fillers.txt")

# Hardcoded commands that should always trigger an interruption
STOP_COMMANDS = {'stop', 'wait', 'hold on', 'shut up', 'silence'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SieveAgent")


class InterruptionLogic:
    """
    Handles the semantic logic for determining if a user's speech input 
    should be classified as a valid interruption or a filler.

    Attributes:
        ignored_words (set): A set of normalized words to be ignored during agent speech.
    """

    def __init__(self):
        """Initializes the logic handler and loads the ignored words list."""
        self.ignored_words = set()
        self.load_ignored_words()

    def _sanitize(self, text):
        """
        Sanitizes input text by removing punctuation and converting to lowercase.

        This ensures that "Uh-uh" and "uh uh" are treated identically.

        Args:
            text (str): The raw transcript text.

        Returns:
            str: The sanitized text string.
        """
        # Efficiently map all punctuation characters to spaces
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        return text.translate(translator).lower()

    def load_ignored_words(self):
        """
        Reads and updates the ignored words set from the external configuration file.
        
        This method is designed to be called repeatedly to support dynamic 
        runtime updates without restarting the agent.
        """
        try:
            if os.path.exists(IGNORED_WORDS_FILE):
                with open(IGNORED_WORDS_FILE, "r", encoding="utf-8") as f:
                    new_words = set()
                    for line in f:
                        # Sanitize line: "Uh-huh" -> "uh huh" -> adds "uh" and "huh"
                        clean_line = self._sanitize(line).strip()
                        if clean_line:
                            new_words.update(clean_line.split())
                    
                    # Only log if the list actually changed to reduce noise
                    if new_words != self.ignored_words:
                        self.ignored_words = new_words
                        logger.info(f"✅ DEBUG: Loaded List: {sorted(list(self.ignored_words))}")
            else:
                # Fallback defaults if file is missing
                logger.warning(f"⚠️ File not found at {IGNORED_WORDS_FILE}. Using defaults.")
                self.ignored_words = {'uh', 'umm', 'hmm', 'haan', 'han', 'ah', 'like', 'right', 'so'}
        except Exception as e:
            logger.error(f"Error reading file: {e}")

    def normalize_word(self, word):
        """
        Normalizes elongated words to their standard form.
        
        Example:
            "hmmmm" -> "hmm"
        
        Args:
            word (str): The individual token to normalize.

        Returns:
            str: The normalized token.
        """
        # Regex looks for a character repeated 3 or more times and reduces it to 2
        return re.sub(r'(.)\1{2,}', r'\1\1', word)

    def is_filler_only(self, text):
        """
        Analyzes a text segment to determine if it consists entirely of filler words.

        Args:
            text (str): The transcript text to analyze.

        Returns:
            bool: True if the text contains *only* fillers, False if it contains *any* valid content.
        """
        # 1. Sanitize (Punctuation -> Space)
        clean_text = self._sanitize(text)
        
        # 2. Split into tokens
        tokens = clean_text.split()
        if not tokens: return False

        # 3. Debug Log to show exactly what we are checking
        logger.info(f"🔍 CHECKING: '{text}' -> Tokens: {tokens}")

        for word in tokens:
            # Check 1: Exact match against the loaded list
            if word in self.ignored_words:
                continue
            
            # Check 2: Normalized match (handles elongated vowels/consonants)
            normalized = self.normalize_word(word)
            if normalized in self.ignored_words:
                continue
            
            # If we find a word NOT in the list, the whole sentence is valid speech
            logger.info(f"❌ VALID WORD FOUND: '{word}' (Not in list)")
            return False

        # All words passed the checks
        return True


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit Worker.
    
    Sets up the connection to the room, initializes the STT plugin, 
    and manages the event loop for processing audio tracks.
    """
    logger.info("Connecting to LiveKit Room...")
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    logger.info(f"User connected: {participant.identity}")

    logic = InterruptionLogic()

    # Initialize Deepgram STT for transcription events
    stt_plugin = deepgram.STT(
        api_key=os.environ.get("DEEPGRAM_API_KEY"),
        model="nova-2-general" 
    )
    
    # Simulation Flag: In a real TTS scenario, this would be bound to the TTS 'playing' state.
    agent_is_speaking = True 
    logger.info(f"⚠️ TEST MODE: Agent speaking state force-set to {agent_is_speaking}")

    async def process_track(track: rtc.Track):
        """
        Handles an incoming audio track, creating streams for STT processing.
        """
        logger.info("🎤 MIC DETECTED - LISTENING...")
        audio_stream = rtc.AudioStream(track)
        stt_stream = stt_plugin.stream()

        async def stt_listener():
            """
            Consumes events from the STT stream and triggers interruption logic.
            """
            async for event in stt_stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    text = event.alternatives[0].text.strip()
                    if not text: continue
                    
                    # Attempt reload on every turn to ensure dynamic updates work immediately
                    logic.load_ignored_words() 
                    
                    # Execute core logic: Distinguish meaningful vs filler
                    handle_interruption_logic(text, logic)

        # Start the listener in the background
        asyncio.create_task(stt_listener())

        # Push audio frames from LiveKit into the STT stream
        async for event in audio_stream:
            if event.frame.data:
                stt_stream.push_frame(event.frame)
    
    def handle_interruption_logic(text, logic_handler):
        """
        Router function that decides the agent's reaction based on the transcript.
        
        Args:
            text (str): The transcribed text.
            logic_handler (InterruptionLogic): The instance of the logic class.
        """
        # Check Commands (sanitize first to catch 'Stop.')
        clean_input = logic_handler._sanitize(text)
        
        # Priority 1: Check for Explicit Stop Commands
        if any(cmd in clean_input for cmd in STOP_COMMANDS):
            # Expected: Agent immediately stops
            logger.warning(f"🔴 INTERRUPT: '{text}' -> STOP.")
            return

        # Priority 2: Check Fillers if the Agent is currently speaking
        if agent_is_speaking:
            # Expected: Agent ignores input if it's just filler
            if logic_handler.is_filler_only(text):
                logger.info(f"🛡️  IGNORED FILLER: '{text}'")
                return
        
        # Priority 3: Register valid speech (interruption or turn-taking)
        logger.info(f"✅ VALID SPEECH: '{text}'")

    # Subscribe to tracks
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(process_track(track))

    # Handle tracks that might already exist upon connection
    for track_pub in participant.track_publications.values():
        if track_pub.track and track_pub.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(process_track(track_pub.track))

    # Keep the worker alive
    await asyncio.Event().wait()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))