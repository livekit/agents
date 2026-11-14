# Copyright 2026 LiveKit, Inc.
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

"""Process warm-up helpers for the Spatius plugin.

At session start the plugin resolves the ingress region through the global
bootstrap API (when ``region="auto"``, the default) and exchanges the API key
for a session token: two sequential HTTPS round trips on the room-join critical
path. Calling ``prewarm`` at the module level of the agent file moves that work to process
start, before a job is dispatched:

```python
from livekit.plugins import spatius

# each job process imports the agent file before it takes a job
spatius.prewarm()
```

The resolved region and the session token are cached process-wide, so
``AvatarSession.start()`` in the dispatched job reuses them. Warm-up is
best-effort: failures are logged and never raised, so process initialization and
the dispatch-time behavior are unaffected.
"""

from __future__ import annotations

import asyncio
import os

from livekit.agents import APIConnectOptions, http

from .avatar import DEFAULT_REGION, _endpoints, _session_token
from .log import logger

# bounded below the worker's default 10-second process initialization timeout
_OPTIONS = APIConnectOptions(max_retry=0, connect_timeout=2.0, read_timeout=2.0)


def prewarm(*, prefetch_session_token: bool = True) -> None:
    """Warm Spatius connection state for this process.

    Call it at the module level of the agent file. Reads the same SPATIUS_*
    environment variables as ``AvatarSession``, resolves and caches the ``auto``
    region and (by default) prefetches a session token, so the first
    ``AvatarSession.start()`` in this process skips both HTTPS round trips.

    Args:
        prefetch_session_token: Prefetch and cache a session token. Assumes the
            Spatius backend allows a token to back more than one session; pass
            False if tokens are single-use.

    Never raises; a failed warm-up just means the dispatch-time path resolves
    and fetches inline as usual.
    """
    app_id = os.getenv("SPATIUS_APP_ID", "")
    if not app_id:
        return  # misconfiguration surfaces in the entrypoint instead

    api_key = os.getenv("SPATIUS_API_KEY", "")
    if prefetch_session_token and not api_key:
        logger.warning("SPATIUS_API_KEY not set; skipping Spatius session-token prefetch")
        prefetch_session_token = False

    try:
        asyncio.run(_prewarm(app_id, api_key if prefetch_session_token else None))
    except Exception:
        # warm-up must never fail process initialization
        logger.warning("Spatius warm-up failed", exc_info=True)


async def _prewarm(app_id: str, api_key: str | None) -> None:
    async with http.open():
        console_url, _ = await _endpoints(
            app_id=app_id,
            region=os.getenv("SPATIUS_REGION", DEFAULT_REGION).strip(),
            console_url=os.getenv("SPATIUS_CONSOLE_ENDPOINT", ""),
            ingress_url=os.getenv("SPATIUS_INGRESS_ENDPOINT", ""),
            conn_options=_OPTIONS,
        )
        if api_key is not None:
            await _session_token(
                api_key=api_key,
                app_id=app_id,
                console_url=console_url,
                conn_options=_OPTIONS,
                prefetch=True,
            )
