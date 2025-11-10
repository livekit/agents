import threading

class SpeechGate:
    """
    Tracks whether the agent TTS is currently speaking.
    Thread-safe boolean latch.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._speaking = False

    def open(self):   # TTS started
        with self._lock:
            self._speaking = True

    def close(self):  # TTS ended
        with self._lock:
            self._speaking = False

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._speaking
