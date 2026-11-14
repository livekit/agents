from __future__ import annotations

import time
from typing import Any


class BackchannelController:
    """Say a short acknowledgment ("ممم") on Nabrah backchannel events.

    Never commits a user turn or calls the LLM. Speaks only while the agent is listening,
    at most once per `cooldown` seconds, and cuts itself off when the user's turn ends.
    """

    def __init__(self, stt: Any, session: Any, *, text: str = "ممم", cooldown: float = 3.0) -> None:
        self.stt, self.session, self.text, self.cooldown = stt, session, text, cooldown
        self._last_at = float("-inf")
        self._handle: Any = None
        stt.on("backchannel", self._on_backchannel)
        stt.on("end_of_turn", self._stop)

    def _busy(self) -> bool:
        return self._handle is not None and not self._handle.done()

    def _stop(self, _event: Any = None) -> None:
        if self._busy():
            self._handle.interrupt()
        self._handle = None

    def _on_backchannel(self, _event: Any) -> None:
        if (
            self.session.agent_state != "listening"
            or self._busy()
            or time.monotonic() - self._last_at < self.cooldown
        ):
            return
        self._handle = self.session.say(self.text, add_to_chat_ctx=False, allow_interruptions=True)
        self._last_at = time.monotonic()

    def close(self) -> None:
        self.stt.off("backchannel", self._on_backchannel)
        self.stt.off("end_of_turn", self._stop)
        self._stop()
