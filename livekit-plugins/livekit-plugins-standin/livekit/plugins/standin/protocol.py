# Copyright 2026 Komaa DigiTech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The StandIn call wire protocol, the subset this plugin speaks.

One WebSocket per live Teams call. JSON text frames, camelCase keys,
discriminated on ``type``. Evolution is additive by contract: unknown fields are
ignored and unknown message types degrade to a no-op, so a newer StandIn and an
older plugin interoperate.

Implemented here (voice + call context):

    in    session.start  audio.frame  ping  participants  dtmf
          recording.status  assistant.say  session.end
    out   audio.frame  pong  session.end

Deliberately not implemented: ``video.frame``, ``expression``, ``speech.marks``,
``display.image`` and ``display.frame``. Those are the avatar surface, which
depends on a third-party avatar runtime and lands separately. Receiving one is
not an error - it falls through the same ignore path as any unknown type.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

#: Both directions are PCM 16 kHz, 16-bit, mono, little-endian.
SAMPLE_RATE_HZ = 16_000
NUM_CHANNELS = 1


@dataclass(frozen=True)
class Caller:
    """Teams caller identity. Every field is optional: Graph returns identities
    without one or more of them for guest and anonymous callers. Blank strings
    are coerced to None so two anonymous callers never collide on an empty AAD
    id and share memory."""

    aad_id: str | None = None
    display_name: str | None = None
    tenant_id: str | None = None


@dataclass(frozen=True)
class SessionStart:
    """First message after the socket opens; the call is live from here."""

    call_id: str
    thread_id: str
    caller: Caller
    direction: str = "inbound"
    recording_status: str | None = None
    #: MANAGED only. Taken from the signed route grant, never caller input, so
    #: it may be trusted to address a chat post during the call.
    tenant_id: str | None = None


def _clean(value: Any) -> str | None:
    """Blank and whitespace-only strings read as absent (see Caller)."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def parse_message(raw: str | bytes) -> dict[str, Any] | None:
    """Decode one wire frame. Returns None for anything that is not a JSON
    object with a string ``type`` - malformed input is dropped, never raised,
    because this runs on the receive path of a live call."""
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("type"), str):
        return None
    return obj


def parse_session_start(msg: dict[str, Any]) -> SessionStart:
    """Read a ``session.start``. Raises ValueError when callId is missing: it is
    the one field with no safe default, since it names the call."""
    call_id = _clean(msg.get("callId"))
    if not call_id:
        raise ValueError("session.start is missing callId")
    raw_caller = msg.get("caller")
    caller_obj = raw_caller if isinstance(raw_caller, dict) else {}
    direction = _clean(msg.get("direction")) or "inbound"
    return SessionStart(
        call_id=call_id,
        # A 1:1 call has no meeting thread, so an absent threadId is normal.
        thread_id=_clean(msg.get("threadId")) or "",
        caller=Caller(
            aad_id=_clean(caller_obj.get("aadId")),
            display_name=_clean(caller_obj.get("displayName")),
            tenant_id=_clean(caller_obj.get("tenantId")),
        ),
        direction=direction if direction in ("inbound", "outbound") else "inbound",
        recording_status=_clean(msg.get("recordingStatus")),
        tenant_id=_clean(msg.get("tenantId")),
    )


def decode_pcm(payload_base64: Any) -> bytes:
    """Decode an ``audio.frame`` payload to raw PCM16 bytes.

    Rejects an odd byte count: PCM16 is 2 bytes per sample, so an odd length
    means a truncated frame. Truncating it silently would shift every following
    sample by a byte and turn the rest of the utterance into noise."""
    if not isinstance(payload_base64, str) or not payload_base64:
        raise ValueError("audio.frame carries no payloadBase64")
    try:
        pcm = base64.b64decode(payload_base64, validate=True)
    except Exception as exc:
        raise ValueError("audio.frame payloadBase64 is not valid base64") from exc
    if len(pcm) < 2 or len(pcm) % 2 != 0:
        raise ValueError(f"malformed PCM16 payload ({len(pcm)} bytes)")
    return pcm


def audio_frame(seq: int, timestamp_ms: int, pcm: bytes) -> str:
    """Build an outbound ``audio.frame``."""
    return json.dumps(
        {
            "type": "audio.frame",
            "seq": seq,
            "timestampMs": timestamp_ms,
            "payloadBase64": base64.b64encode(pcm).decode("ascii"),
        },
        separators=(",", ":"),
    )


def pong(ts: Any) -> str:
    """Answer a ``ping``. The timestamp is echoed verbatim so the worker can
    measure the round trip against its own clock."""
    return json.dumps(
        {"type": "pong", "ts": ts if isinstance(ts, int) else 0}, separators=(",", ":")
    )


def session_end(reason: str) -> str:
    """Advisory teardown notice. The close that follows is what ends the call;
    this carries the only answer to "why" the worker cannot derive itself."""
    return json.dumps({"type": "session.end", "reason": reason}, separators=(",", ":"))
