import asyncio
import logging
import time
import traceback
import os
from typing import AsyncIterable, AsyncGenerator, Any
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    stt,
)
from livekit.agents.llm import ChatContext, ChatMessage, function_tool
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# --- Configuration ---
# Set up logging to see our custom logs
# This will make our "GATEKEEPER" logs visible in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("robust-agent")
load_dotenv()

# --- V13 DYNAMIC CONFIGURATION (Bonus Challenge 2: Multi-language) ---
# Load default filler words from an environment variable.
# This list can contain multi-language fillers (e.g., English + Hindi).
# This is your preferred list:
DEFAULT_FILLERS_CSV = "um,umm,uh,hmm,hmmm,haan,han,achha,acha,ok,okay,like,yeah,right,a,i"

# Read from .env file (or system environment)
fillers_csv = os.environ.get("IGNORED_FILLERS_CSV", DEFAULT_FILLERS_CSV)

# Parse the comma-separated string into a set for fast lookups
INITIAL_FILLERS = {filler.strip().lower() for filler in fillers_csv.split(',') if filler.strip()}
logger.info(f"Loaded {len(INITIAL_FILLERS)} default filler words: {INITIAL_FILLERS}")
# --- END DYNAMIC CONFIGURATION ---


class RobustAgent(Agent):
    """
    A smart, context-aware agent that filters interruptions.
    
    It overrides the 'stt_node' to create a "gatekeeper" that buffers
    speech signals. It checks if the agent is currently speaking:
    - If YES: It ignores fillers and noise.
    - If NO: It allows all speech to pass through immediately.
    
    It also includes function tools to dynamically update the filler list
    at runtime (Bonus Challenge 1).
    """
    def __init__(self):
        # --- V14 BONUS CHALLENGE 1: DYNAMIC RUNTIME UPDATE ---
        # Store fillers on the instance, loading from our initial list
        self.ignored_fillers = INITIAL_FILLERS.copy()

        # Define the dynamic update tools
        tools = [
            self.add_ignored_filler,
            self.remove_ignored_filler,
        ]
        # --- END V14 ---

        super().__init__(
            instructions=(
                "You are a helpful and concise assistant. Keep answers short. "
                "You have two tools: 'add_ignored_filler' and 'remove_ignored_filler'. "
                "If a user asks you to 'please ignore [word]' or 'stop ignoring [word]', "
                "you MUST call the appropriate tool."
            ),
            tools=tools, # Pass tools to the base Agent
        )
        
    # --- V14 BONUS CHALLENGE 1: DYNAMIC UPDATE FUNCTIONS ---
    @function_tool
    async def add_ignored_filler(self, word: str):
        """
        Dynamically adds a new word to the list of fillers to ignore.
        Use this if a user complains about a specific word causing interruptions.
        """
        clean_word = word.strip().lower()
        if clean_word not in self.ignored_fillers:
            self.ignored_fillers.add(clean_word)
            logger.info(f"Dynamically ADDED filler: '{clean_word}'. Total: {len(self.ignored_fillers)}")
            return f"Okay, I will now ignore the filler word '{clean_word}'."
        return f"I am already ignoring the word '{clean_word}'."

    @function_tool
    async def remove_ignored_filler(self, word: str):
        """
        Dynamically removes a word from the ignored filler list.
        Use this if a word (e.g., 'okay') is being ignored too often.
        """
        clean_word = word.strip().lower()
        if clean_word in self.ignored_fillers:
            self.ignored_fillers.remove(clean_word)
            logger.info(f"Dynamically REMOVED filler: '{clean_word}'. Total: {len(self.ignored_fillers)}")
            return f"Okay, I will no longer ignore the word '{clean_word}'."
        return f"I was not ignoring the word '{clean_word}' anyway."
    # --- END V14 ---

    def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        """
        This is the override that intercepts the audio-to-text stream.
        We return our custom filter instead of the default one.
        """
        original_stream = super().stt_node(audio, model_settings)
        return self._filtered_stt_stream(original_stream)

    def _is_speaking(self) -> bool:
        """
        Helper to check if the agent is currently playing audio.
        This is the "context-aware" part of the logic.
        """
        try:
             # Access the SESSION's public 'current_speech' property.
             if self.session and self.session.current_speech:
                 return not self.session.current_speech.done()
        except Exception as e:
            # This can fail if session isn't fully set up, default to not speaking
            pass
        return False

    async def _filtered_stt_stream(
        self, upstream: AsyncGenerator[stt.SpeechEvent, None]
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        """
        This is the core filter logic.
        """
        buffered_start_event = None
        interruption_start_time = 0.0
        has_seen_valid_text = False
        
        async for event in upstream:
            # --- 1. User started making noise ---
            if event.type == stt.SpeechEventType.START_OF_SPEECH:
                if self._is_speaking():
                    # AGENT IS SPEAKING: Engage the filter.
                    interruption_start_time = time.time()
                    has_seen_valid_text = False
                    logger.info("🔒 GATEKEEPER: Agent is speaking. Holding audio gate closed...")
                    buffered_start_event = event
                    continue # Don't yield yet
                else:
                    # AGENT IS SILENT: Let the event pass through immediately.
                    logger.info("🟢 PASS-THROUGH: Agent is silent. Allowing user to speak.")
                    yield event
                    continue

            # --- 2. User speech was transcribed ---
            elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT or \
                 event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                
                if not event.alternatives or not event.alternatives[0].text:
                    if not buffered_start_event: yield event
                    continue

                raw_text = event.alternatives[0].text.strip()
                clean_text = raw_text.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").strip()

                if buffered_start_event:
                    # We are in "filter mode" because the agent was speaking.
                    duration = time.time() - interruption_start_time
                    
                    # Check the instance's dynamic list
                    is_filler = clean_text in self.ignored_fillers or clean_text == ""
                    
                    if self._is_speaking() and is_filler:
                         # Agent is still talking AND it's a filler. Ignore it.
                         has_seen_valid_text = True 
                         logger.info(f"🛡️ IGNORED FILLER (+{duration:.2f}s): '{clean_text}' -> Agent continues.")
                         continue
                    else:
                        # It's real speech OR agent finished talking. Open the gate!
                        reason = "Real speech" if not is_filler else "Agent stopped naturally"
                        logger.info(f"🔓 GATE OPEN (+{duration:.2f}s): {reason} '{clean_text}' -> INTERRUPTING!")
                        yield buffered_start_event # This will now trigger the interruption
                        buffered_start_event = None
                        yield event
                else:
                    # We are in "pass-through" mode. Let it flow.
                    yield event

            # --- 3. User stopped speaking ---
            elif event.type == stt.SpeechEventType.END_OF_SPEECH:
                if buffered_start_event:
                    # User stopped talking, but we never opened the gate.
                    # This means they *only* said fillers.
                    duration = time.time() - interruption_start_time
                    if has_seen_valid_text:
                         logger.info(f"❌ DISCARD: Ignored sequence of fillers ({duration:.2f}s) while agent spoke.")
                    else:
                         logger.info(f"🔉 DISCARD: Ignored pure noise ({duration:.2f}s) while agent spoke.")
                    buffered_start_event = None
                    continue
                yield event
            else:
                # Pass all other event types (like metrics)
                yield event

async def entrypoint(ctx: JobContext):
    logger.info("Agent entrypoint called.")
    
    # --- THE CRITICAL "VADECTOMY" SETUP ---
    # We set vad=None to disable the "dumb" VAD interruption path.
    # Our RobustAgent's stt_node is now the *only* source of interruption signals.
    session = AgentSession(
        vad=None,                          # <--- THIS IS THE KEY
        turn_detection=MultilingualModel(),# Keeps knowing when you finish a turn
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        allow_interruptions=True,          # We allow interruptions, but only WE trigger them
    )
    
    try:
        await ctx.connect()
        logger.info("Agent connected to room.")

        agent = RobustAgent()
        await session.start(agent=agent, room=ctx.room)
        logger.info("RobustAgent session started.")
        
        # Greet the user to start the interaction
        await session.generate_reply(
            instructions="Greet the user and ask them a question. Tell them to try interrupting you with 'umm' or 'haan'."
        )
        logger.info("Initial greeting queued.")

    except Exception as e:
        logger.error(f"Error in entrypoint: {e}", exc_info=True)
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("Starting agent worker...")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))