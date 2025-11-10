import os
import time
from typing import Set, Dict, List


class FillerManager:
    """
    Loads filler words from a directory and auto-updates them at runtime.
    Supports:
    - English fillers
    - Hindi fillers
    - Hinglish variants
    - Custom fillers
    - Command words (stop, wait, rukko, bas, etc.)
    """

    def __init__(self, folder: str):
        self.folder = folder
        self.last_loaded_time = 0
        self.fillers: Set[str] = set()
        self.commands: Set[str] = set()
        self._file_cache: Dict[str, float] = {}

        if not os.path.isdir(folder):
            raise RuntimeError(f"Filler folder not found: {folder}")

        self.reload()  # Initial load

    # -----------------------------------------
    # File Helpers
    # -----------------------------------------

    def _read_txt(self, path: str) -> Set[str]:
        if not os.path.exists(path):
            return set()

        try:
            with open(path, "r", encoding="utf-8") as f:
                return {
                    line.strip().lower()
                    for line in f
                    if line.strip()
                }
        except Exception:
            return set()

    def _detect_changes(self) -> bool:
        """
        Returns True if any .txt file changed (mtime updated).
        """
        changed = False

        for fname in os.listdir(self.folder):
            if not fname.endswith(".txt"):
                continue

            path = os.path.join(self.folder, fname)
            try:
                mtime = os.path.getmtime(path)
            except FileNotFoundError:
                continue

            if path not in self._file_cache or mtime > self._file_cache[path]:
                self._file_cache[path] = mtime
                changed = True

        return changed

    # -----------------------------------------
    # Main Reload Logic
    # -----------------------------------------

    def reload(self):
        """
        Reloads fillers & commands if any file changed.
        """
        if not self._detect_changes():
            return  # No update needed

        new_fillers = set()
        new_commands = set()

        for fname in os.listdir(self.folder):
            if not fname.endswith(".txt"):
                continue

            path = os.path.join(self.folder, fname)
            words = self._read_txt(path)

            if "command" in fname.lower():
                new_commands |= words
            else:
                new_fillers |= words

        self.fillers = new_fillers
        self.commands = new_commands
        self.last_loaded_time = time.time()

        print(
            f"[Dynamic Reload] Updated fillers={len(self.fillers)} commands={len(self.commands)}"
        )

    # -----------------------------------------
    # Query Helpers
    # -----------------------------------------

    def is_filler(self, word: str) -> bool:
        return word.lower() in self.fillers

    def is_command(self, word: str) -> bool:
        return word.lower() in self.commands


# --------------------------------------------------------
# Hinglish / Hindi / English normalization utilities
# --------------------------------------------------------

def normalize_words(words: List[str]) -> List[str]:
    """
    Normalize Hindi-English mixed speech:
    - Lowercase
    - Strip punctuation
    - Map variants to canonical form
    """

    mapping = {
        # Hindi/Hinglish
        "haan": "haan",
        "han": "haan",
        "haanji": "haan",
        "haan ji": "haan",
        "haanji?": "haan",
        "accha": "accha",
        "acha": "accha",
        "achha": "accha",
        "theek": "thik",
        "thik": "thik",

        # English fillers
        "ok": "okay",
        "okk": "okay",
        "okayyy": "okay",
        "umm": "um",
        "uhh": "uh",
        "hmm ok": "hmm okay",
        "hmm okay": "hmm okay",
        "hmmkay": "hmm okay",
    }

    normalized = []
    for w in words:
        w = w.lower().strip(" ,.!?;:-\"'()")
        normalized.append(mapping.get(w, w))

    return normalized
