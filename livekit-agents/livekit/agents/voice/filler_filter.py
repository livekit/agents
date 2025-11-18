class FillerFilter:
    FILLERS = {
        "uh", "um", "hmm", "er", "ah", "eh",
        "okay", "ok", "huh", "oh", "hmm", "right"
    }

    def is_filler_only(self, text: str) -> bool:
        if not text.strip():
            return True

        words = [w.lower().strip(".,!?") for w in text.split()]
        return all(w in self.FILLERS for w in words)

    def is_low_confidence(self, confidence: float, threshold: float = 0.50) -> bool:
        return confidence < threshold

