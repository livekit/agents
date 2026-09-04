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

"""The StandIn handshake HMAC, both directions.

One construction everywhere: ``HMAC-SHA256(secret, "{timestampMs}.{id}")``,
lowercase hex, carried in ``X-StandIn-Timestamp`` / ``X-StandIn-Signature``.

    inbound   StandIn dials the call listener; ``id`` is the callId in the URL
              path, and the plugin VERIFIES inside a replay window.
    outbound  the worker dials the chat channel; ``id`` is the channel name,
              and the plugin SIGNS.
"""

from __future__ import annotations

import hashlib
import hmac
import time

TIMESTAMP_HEADER = "X-StandIn-Timestamp"
SIGNATURE_HEADER = "X-StandIn-Signature"

#: Handshakes are dialed and answered immediately; anything older is a replay.
REPLAY_WINDOW_MS = 60_000


def now_ms() -> int:
    return int(time.time() * 1000)


def sign_handshake(secret: str, timestamp_ms: int | str, handshake_id: str) -> str:
    """Signature for a WebSocket upgrade."""
    payload = f"{timestamp_ms}.{handshake_id}".encode()
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_handshake(
    secret: str,
    timestamp: str | None,
    handshake_id: str,
    signature: str | None,
    current_ms: int | None = None,
) -> bool:
    """Constant-time check of an inbound upgrade. Empty inputs fail CLOSED."""
    if not secret or not timestamp or not signature:
        return False
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs((now_ms() if current_ms is None else current_ms) - ts) > REPLAY_WINDOW_MS:
        return False
    expected = sign_handshake(secret, timestamp, handshake_id)
    return hmac.compare_digest(expected, signature.strip().lower())
