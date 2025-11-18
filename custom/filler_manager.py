import json
import os
import re
import time


class FillerManager:
    """
    Loads and manages filler words.
    Supports:
    - English + Hindi filler detection
    - Dynamic updates (reload JSON at runtime)
    - Pure filler detection
    - Mixed filler + normal speech detection
    """

    def __init__(self, filler_file_path, reload_interval=60):
        """
        filler_file_path  : path to fillers_default.json
        reload_interval   : time in seconds to auto-reload filler list
        """
        self.filler_file_path = filler_file_path
        self.reload_interval = reload_interval
        self.last_reload_time = 0

        self.fillers = []
        self.load_fillers()

        # Fallback heuristics
        self.short_filler_pattern = re.compile(r"^(h*m+|u+h*|a+h*|h+)$", re.IGNORECASE)

    # -------------------------------------------------------
    # Load fillers from JSON
    # -------------------------------------------------------
    def load_fillers(self):
        """Load filler words from JSON file."""
        try:
            with open(self.filler_file_path, "r") as f:
                self.fillers = [w.lower().strip() for w in json.load(f)]
        except Exception:
            self.fillers = ["uh", "umm", "hmm", "haan", "haanji", "arey", "acha"]

        self.last_reload_time = time.time()

    # -------------------------------------------------------
    # Dynamic update (bonus requirement)
    # -------------------------------------------------------
    def maybe_reload_fillers(self):
        """Reload JSON file automatically every reload_interval."""
        if time.time() - self.last_reload_time >= self.reload_interval:
            self.load_fillers()

    def update_fillers(self, new_list):
        """Update filler list dynamically."""
        self.fillers = [w.lower().strip() for w in new_list]
        self.last_reload_time = time.time()

    # -------------------------------------------------------
    # Check if text is PURE filler
    # -------------------------------------------------------
    def is_filler(self, text):
        """
        Returns True if the text is only filler words.
        """
        self.maybe_reload_fillers()

        words = text.split()

        if not words:
            return False

        # ALL words must be filler
        for w in words:
            if w not in self.fillers:
                # Check heuristic for sound-like fillers
                if not self.short_filler_pattern.match(w):
                    return False

        return True

    # -------------------------------------------------------
    # Check if text contains ANY filler words (mixed speech)
    # -------------------------------------------------------
    def contains_filler(self, text):
        """
        Returns True if ANY filler word appears in the text.
        """
        self.maybe_reload_fillers()

        words = text.split()
        for w in words:
            if w in self.fillers:
                return True
        return False
