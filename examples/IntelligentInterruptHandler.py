# examples/IntelligentInterruptHandler.py

from typing import List

class IntelligentInterruptHandler:
    """
    An extension layer to intelligently distinguish meaningful user interruptions 
    from irrelevant fillers using ASR results, ensuring no changes to LiveKit's 
    base VAD algorithm[cite: 7, 12].
    """
    
    def __init__(self, ignored_words: List[str], interruption_commands: List[str], confidence_threshold: float):
        # Store configuration lists, ensuring they are lowercase for comparison
        self.ignored_words = set(w.lower() for w in ignored_words)
        self.interruption_commands = set(w.lower() for w in interruption_commands)
        self.confidence_threshold = confidence_threshold
        print("IntelligentInterruptHandler initialized.")

    def _get_clean_words(self, transcript: str) -> List[str]:
        # Simple tokenization: converts to lowercase and splits the transcript
        return transcript.lower().split()

    def should_interrupt(self, transcript: str, confidence: float, agent_is_speaking: bool) -> bool:
        """
        Determines the interruption decision based on the current agent state and transcript content.
        
        Args:
            transcript: The ASR result text (e.g., "umm okay stop").
            confidence: The ASR confidence score (used for background murmur check).
            agent_is_speaking: True if the agent is currently sending TTS audio.
            
        Returns:
            True if the agent should stop immediately [cite: 18], False otherwise[cite: 16].
        """
        words = self._get_clean_words(transcript)
        
        if not words:
            return False

        # --- High-Priority Command Check ---
        # If genuine user speech (like "wait" or "stop") occurs, the agent must stop immediately[cite: 11, 18].
        if any(word in self.interruption_commands for word in words):
            print(f"**LOG: VALID COMMAND INTERRUPTION** - Command '{transcript}' detected.")
            # Expected: Agent immediately stops (handles "umm okay stop" scenario)[cite: 21].
            return True

        # --- Agent Speaking Logic ---
        if agent_is_speaking:
            
            # Scenario: Background murmur (low confidence ASR)[cite: 22].
            if confidence < self.confidence_threshold:
                print(f"**LOG: IGNORED LOW CONFIDENCE** - Confidence {confidence:.2f} < Threshold.")
                return False
                
            # Scenario: User filler while agent speaks[cite: 15].
            is_filler_only = all(word in self.ignored_words for word in words)
            if is_filler_only:
                print(f"**LOG: IGNORED FILLER** - Agent speaking, ignoring filler: '{transcript}'.")
                # Expected: Agent ignores input and continues speaking[cite: 16, 57].
                return False
            
            # Scenario: Any other meaningful speech while agent speaks is a real interruption.
            print(f"**LOG: VALID SPEECH INTERRUPTION** - Agent speaking, stopping for meaningful speech: '{transcript}'.")
            return True
        
        # --- Agent Quiet Logic ---
        else:
            # Scenario: User filler while agent quiet[cite: 20].
            # When the agent is quiet, any speech registers as a valid user turn[cite: 10].
            print(f"**LOG: SPEECH EVENT** - Agent quiet, registering speech: '{transcript}'.")
            # Returning True here is mostly for logging/debugging; the core VAD logic handles the turn start.
            return True