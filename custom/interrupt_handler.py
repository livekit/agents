class InterruptDecision:
    IGNORE = "ignore"
    INTERRUPT = "interrupt"
    NORMAL = "normal"


class InterruptHandler:
    def __init__(self, filler_manager, command_list, confidence_threshold=0.6):
        self.filler_manager = filler_manager
        self.command_list = [c.lower().strip() for c in command_list]
        self.confidence_threshold = confidence_threshold

    def process_transcript(self, text, confidence, agent_speaking):
        text = text.lower().strip()

        # -------------------------------
        # 1️⃣ Command detection (highest priority)
        # -------------------------------
        if self._is_command(text):
            return InterruptDecision.INTERRUPT

        # -------------------------------
        # 2️⃣ Low confidence → ignore if filler
        # -------------------------------
        if confidence < self.confidence_threshold:
            if self.filler_manager.is_filler(text):
                return InterruptDecision.IGNORE

        # -------------------------------
        # 3️⃣ When agent is speaking
        # -------------------------------
        if agent_speaking:

            # If input is a filler → ignore
            if self.filler_manager.is_filler(text):
                return InterruptDecision.IGNORE

            # If input is meaningful speech → interrupt
            return InterruptDecision.INTERRUPT

        # -------------------------------
        # 4️⃣ When agent is NOT speaking
        # -------------------------------
        return InterruptDecision.NORMAL

    # -----------------------------------------------------
    # Command detection must be VERY robust
    # -----------------------------------------------------
    def _is_command(self, text):
        for cmd in self.command_list:
            # Exact match
            if text == cmd:
                return True

            # Match beginning (e.g., "stop now", "wait please")
            if text.startswith(cmd + " "):
                return True

            # Match with punctuation (e.g., "stop.", "wait!")
            if text.startswith(cmd):
                return True

        return False
