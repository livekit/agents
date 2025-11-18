class StateTracker:
    """
    Tracks whether the agent is currently speaking.
    If speaking → user speech may interrupt.
    If silent → everything is normal 'speech'.
    """

    def __init__(self):
        self._is_speaking = False

    def set_speaking(self, speaking: bool):
        self._is_speaking = speaking

    def is_speaking(self) -> bool:
        return self._is_speaking