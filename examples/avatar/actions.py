"""Avatar pose triggers via LLM tools (wave, dance, turn)."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger("avatar.actions")

NONE = "none"

ACTION_PERSONAS = frozenset({"leila", "jess", "mr_fox"})

POSE_NAMES: dict[str, dict[str, str]] = {
    "leila": {
        "wave": "wave-2-leila",
        "turn": "turn-leila",
        "dance": "dance-leila",
    },
    "jess": {
        "wave": "jess_wave",
        "turn": "jess_turn",
        "dance": "jess_dance",
    },
    "mr_fox": {
        "wave": "fox2_wave",
        "turn": "fox2_turn",
        "dance": "fox2_dance",
    },
}

DEFAULT_POSE_DURATION_S = 6.0
OPENING_WAVE_DELAY_S = 0.5


def supports_actions(persona_id: str) -> bool:
    return persona_id in ACTION_PERSONAS


def _control_url(session_id: str) -> str:
    base = os.getenv("LEMONSLICE_API_BASE", "https://lemonslice.com/api").rstrip("/")
    return f"{base}/liveai/sessions/{session_id}/control"


async def trigger_pose(session_id: str, name: str) -> bool:
    url = _control_url(session_id)
    payload = {"event": "pose-trigger", "pose_trigger": {"name": name}}
    timeout = aiohttp.ClientTimeout(total=30.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            url,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": os.environ["LEMONSLICE_API_KEY"],
            },
            json=payload,
        ) as response:
            return response.ok


@dataclass
class _PlayingSlot:
    ends_at: float


class ActionController:
    """Plays one LemonSlice pose at a time; each blocks others for ``DEFAULT_POSE_DURATION_S``."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._session_id: str | None = None
        self._persona_id: str | None = None
        self._slot: _PlayingSlot | None = None

    def set_session(self, session_id: str, persona_id: str) -> None:
        self._session_id = session_id
        self._persona_id = persona_id

    def clear_session(self) -> None:
        self._session_id = None
        self._persona_id = None

    def _current_slot(self) -> _PlayingSlot | None:
        if self._slot is None:
            return None
        if time.monotonic() >= self._slot.ends_at:
            self._slot = None
            return None
        return self._slot

    async def cancel(self) -> None:
        async with self._lock:
            self._slot = None
        sid = self._session_id
        self.clear_session()
        if sid is not None:
            await trigger_pose(sid, NONE)

    async def shutdown(self, _: str = "") -> None:
        await self.cancel()

    async def play(self, action_id: str) -> str:
        session_id = self._session_id
        persona_id = self._persona_id
        if session_id is None or persona_id is None:
            return "Motion unavailable — avatar session not ready."

        key = action_id.strip().lower()
        pose_name = POSE_NAMES.get(persona_id, {}).get(key)
        if pose_name is None:
            return f"Unknown motion {action_id!r}."

        async with self._lock:
            if self._current_slot() is not None:
                return "That motion is already playing; try again in a moment."

            ok = await trigger_pose(session_id, pose_name)
            if not ok:
                return "Could not trigger the motion on the avatar."

            self._slot = _PlayingSlot(
                ends_at=time.monotonic() + DEFAULT_POSE_DURATION_S,
            )
            logger.info(
                "pose playing: persona_id=%r action_id=%r pose_name=%r",
                persona_id,
                key,
                pose_name,
            )
            return f"Playing motion {key}."

    async def opening_wave(self) -> None:
        if OPENING_WAVE_DELAY_S > 0:
            await asyncio.sleep(OPENING_WAVE_DELAY_S)
        await self.play("wave")
