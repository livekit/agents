import re
import os
import json
import asyncio
from dotenv import load_dotenv

class InterruptHandler:
    """
    A class to intelligently distinguish between filler words and
    genuine user interruptions during a LiveKit voice conversation.
    Supports configurable filler lists via .env or JSON file.
    """

    def __init__(self, ignored_words=None, confidence_threshold=0.6):
        # Load environment variables
        load_dotenv()

        # 1️⃣ Default filler words (English + Hinglish)
        default_fillers = {
            # ===== English Fillers =====
            "uh", "umm", "hmm", "hm", "huh", "ah", "oh", "er", "uhh", "uhmm", "mmm", "mm",
            "hmmm", "like", "you know", "i mean", "actually", "basically", "so yeah", "okay",
            "ok", "alright", "right", "yeah", "yep", "yup", "yaa", "yuh",

            # ===== Hindi / Hinglish Fillers =====
            "haan", "haina", "arey", "accha", "acha", "arre", "matlab", "toh", "yaar",
            "bas", "theek hai", "achha", "thik hai", "haan na", "haan ji", "haan okay",
            "hmmm haan", "haan theek hai", "acha theek hai", "hmm theek hai", "haan na bhai",
            "haan sahi", "ha bhai", "ha theek", "ha okay", "haan haan", "haan bilkul",

            # ===== Multilingual / Mixed casual phrases =====
            "okay okay", "ok ok", "hmm okay", "huh okay", "haan okay", "hmm haan",
            "theek theek", "hmm okayy", "accha haan", "hmm accha", "hmmm okay",
            "haan hmm", "haan ha", "acha haan", "hmm ha", "arre haan", "accha theek",
            "okayy", "okk", "okie", "okiee", "okies", "hmm hmm",

            # ===== Fillers with partial affirmations =====
            "yeah yeah", "right right", "okay right", "hmm yeah", "uh yeah", "ok fine",
            "yep yep", "ok then", "okey dokey", "oh okay", "ohk", "ok fine", "ok sure"
        }


        # 2️⃣ Try loading fillers from .env first
        env_fillers = os.getenv("IGNORED_WORDS")
        if env_fillers:
            print("[CONFIG] Loading ignored words from .env file...")
            try:
                self.ignored_words = {w.strip().lower() for w in env_fillers.split(",") if w.strip()}
            except Exception as e:
                print(f"[WARN] Could not parse IGNORED_WORDS from .env: {e}")
                self.ignored_words = default_fillers
        else:
            # 3️⃣ Optional: load from JSON if available
            json_path = os.getenv("IGNORED_WORDS_JSON", "ignored_words.json")
            if os.path.exists(json_path):
                print(f"[CONFIG] Loading ignored words from {json_path}...")
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # Flatten any nested structure (dict/lists)
                        if isinstance(data, dict):
                            all_words = set()
                            for v in data.values():
                                all_words |= set(map(str.lower, v))
                            self.ignored_words = all_words
                        elif isinstance(data, list):
                            self.ignored_words = set(map(str.lower, data))
                        else:
                            raise ValueError("Invalid JSON structure for ignored words.")
                except Exception as e:
                    print(f"[WARN] Failed to load JSON file: {e}")
                    self.ignored_words = default_fillers
            else:
                self.ignored_words = default_fillers

        self.confidence_threshold = confidence_threshold
        self.agent_speaking = False
        self.logs = []
        print(f"[INIT] Loaded {len(self.ignored_words)} filler words.")

    def set_agent_state(self, speaking: bool):
        """Called whenever the agent starts or stops speaking."""
        self.agent_speaking = speaking
        print(f"[STATE] Agent speaking: {self.agent_speaking}")

    async def handle_transcript(self, text: str, confidence: float):
        """
        Handle each transcript event (ASR result).
        Returns text if it's a real interruption,
        or None if it should be ignored.
        """
        text = text.strip().lower()

        # 1️⃣ Ignore empty or low-confidence transcriptions
        if not text or confidence < self.confidence_threshold:
            self.logs.append(("ignored_low_conf", text))
            print(f"[IGNORED: low_conf] '{text}' (conf={confidence:.2f})")
            return None

        # 2️⃣ If agent is speaking, filter out filler-only speech
        if self.agent_speaking:
            # Normalize repeating letters for fuzzy matching
            normalized = re.sub(r'(.)\1{2,}', r'\1', text)  # e.g. 'ummm' → 'um', 'okayy' → 'okay'

            # If normalized phrase matches ignored fillers
            if normalized in self.ignored_words:
                self.logs.append(("ignored_fuzzy_filler", text))
                print(f"[IGNORED: fuzzy filler] '{text}' → '{normalized}'")
                return None

            words = re.findall(r'\b\w+\b', text)
            if all(word in self.ignored_words for word in words):
                self.logs.append(("ignored_filler", text))
                print(f"[IGNORED: filler] '{text}'")
                return None
            if len(words) == 1 and words[0] in self.ignored_words:
                self.logs.append(("ignored_single_word", text))
                print(f"[IGNORED: single filler] '{text}'")
                return None

        # 3️⃣ Otherwise, treat as valid interruption
        self.logs.append(("valid_interrupt", text))
        print(f"[INTERRUPT] '{text}'")
        return text

    def add_ignored_word(self, word: str):
        """Dynamically add a filler word at runtime."""
        word = word.strip().lower()
        if word not in self.ignored_words:
            self.ignored_words.add(word)
            print(f"[CONFIG] Added ignored word: '{word}'")
        else:
            print(f"[CONFIG] Word already in ignored list: '{word}'")
