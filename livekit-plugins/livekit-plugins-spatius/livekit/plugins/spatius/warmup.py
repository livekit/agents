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

At session start the Spatius SDK resolves the ingress region through the global
bootstrap API (when ``region="auto"``, the default) and exchanges the API key
for a session token — two sequential HTTPS round trips on the room-join critical
path. Wiring ``prewarm`` into the worker moves that work to process
initialization, before a job is dispatched:

```python
from livekit import agents
from livekit.plugins import spatius


def prewarm(proc: agents.JobProcess) -> None:
    spatius.prewarm(proc)


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )
```

The spatius SDK caches the resolved region and the session token process-wide,
so ``AvatarSession.start()`` in the dispatched job reuses them. Warm-up is
best-effort: failures are logged and never raised, so process initialization and
the dispatch-time behavior are unaffected.
"""

from __future__ import annotations

import asyncio
import os

from livekit.agents import JobProcess

from .log import logger

# keep in sync with the environment variables read by AvatarSession
_ENV_API_KEY = "SPATIUS_API_KEY"
_ENV_APP_ID = "SPATIUS_APP_ID"
_ENV_REGION = "SPATIUS_REGION"
_ENV_CONSOLE_ENDPOINT = "SPATIUS_CONSOLE_ENDPOINT"
_ENV_INGRESS_ENDPOINT = "SPATIUS_INGRESS_ENDPOINT"

# "auto" sentinel, mirrors spatius.session_config.DEFAULT_REGION_REQUEST
_AUTO_REGION = "auto"


def prewarm(proc: JobProcess, *, prefetch_session_token: bool = True) -> None:
    """Warm Spatius connection state for this job process.

    Pass to ``WorkerOptions(prewarm_fnc=...)``. Reads the same SPATIUS_*
    environment variables as ``AvatarSession`` and delegates to
    ``spatius.prewarm()``, which resolves and caches the ``auto`` region, warms
    TLS to the console and ingress endpoints, and (by default) prefetches a
    session token so the first ``AvatarSession.start()`` in this process skips
    both HTTPS round trips.

    Args:
        proc: The job process being initialized.
        prefetch_session_token: Prefetch and cache a session token. Assumes the
            Spatius backend allows a token to back more than one session; pass
            False if tokens are single-use.

    Never raises; a failed warm-up just means the dispatch-time path resolves
    and fetches inline as usual.
    """
    app_id = os.getenv(_ENV_APP_ID, "")
    if not app_id:
        return  # misconfiguration surfaces in the entrypoint instead

    api_key = os.getenv(_ENV_API_KEY, "")
    if prefetch_session_token and not api_key:
        logger.warning("SPATIUS_API_KEY not set; skipping Spatius session-token prefetch")
        prefetch_session_token = False

    import spatius

    try:
        result = asyncio.run(
            spatius.prewarm(
                app_id=app_id,
                api_key=api_key or None,
                region=os.getenv(_ENV_REGION, _AUTO_REGION),
                console_endpoint_url=os.getenv(_ENV_CONSOLE_ENDPOINT, ""),
                ingress_endpoint_url=os.getenv(_ENV_INGRESS_ENDPOINT, ""),
                prefetch_session_token=prefetch_session_token,
            )
        )
    except Exception:
        # warm-up must never fail process initialization
        logger.warning("Spatius warm-up failed", exc_info=True)
        return

    logger.debug(
        "Spatius warm-up finished",
        extra={
            "region": result.region,
            "tls_warmed": result.tls_warmed,
            "session_token_prefetched": result.session_token_prefetched,
        },
    )
