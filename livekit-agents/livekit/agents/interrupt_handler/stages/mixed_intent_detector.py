import os
import re

class MixedIntentDetector:
    """
    Command intent detector:
    - exact command words
    - multiword phrases ("hold on", "change topic")
    - fuzzy match ("stooop", "waaait")
    - ENV configurable keyword list
    - language-agnostic
    """

    def __init__(self, command_keywords=None):
        # ENV override
        env_cmds = os.getenv("COMMAND_WORDS")
        if env_cmds:
            command_keywords = [w.strip() for w in env_cmds.split(",")]

        # defaults
        self.commands = [cmd.lower() for cmd in (command_keywords or [
            "stop", "wait", "pause",
            "hold on", "change topic",
            "no", "stop now"
        ])]

        # Precompile regular expressions for fast fuzzy matching
        self.command_patterns = []
        for cmd in self.commands:
            # fuzzy: allow repeated characters ("stoooop", "waaaait")
            fuzzy = "".join(
                f"{re.escape(c)}+" if c.isalpha() else re.escape(c)
                for c in cmd
            )
            pattern = re.compile(rf"\b{fuzzy}\b")
            self.command_patterns.append(pattern)

    def contains_command(self, transcript: str) -> bool:
        text = transcript.lower().strip()

        # Direct phrase detection
        for cmd in self.commands:
            if cmd in text:
                return True

        # Fuzzy match detection
        for pattern in self.command_patterns:
            if pattern.search(text):
                return True

        return False