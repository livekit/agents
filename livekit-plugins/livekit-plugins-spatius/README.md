# LiveKit Plugins Spatius

Agent Framework plugin for [Spatius](https://www.spatius.ai/?utm_source=livekit) avatars.

See the [Spatius documentation](https://docs.spatius.ai?utm_source=livekit) for Spatius account setup and
avatar configuration.

## Client-side rendering

Spatius avatars are rendered on the client instead of being sent as conventional
server-rendered video. The avatar's LiveKit video track carries motion data in
otherwise black frames, so a standard LiveKit video renderer will display a black
screen. Your frontend must use the Spatius client SDK and LiveKit adapter to decode
the track and render the avatar.

See the [client integration guide](https://docs.spatius.ai/livekit-agents/client?utm_source=livekit) and
the [reference frontend](https://github.com/spatius-ai/spatius-avatar-demo/tree/main/platform-integrations/livekit-agents-demo/livekit-agents-reference-demo/frontend)
for a working implementation.

## Installation

```bash
pip install livekit-plugins-spatius
```

## Usage

```python
from livekit.plugins import spatius

avatar = spatius.AvatarSession()
await avatar.start(session, room=ctx.room)
```

The plugin reads `SPATIUS_API_KEY`, `SPATIUS_APP_ID`, and `SPATIUS_AVATAR_ID` from the
environment when constructor arguments are omitted.

## Warm-up

At session start, the Spatius SDK resolves the ingress region via the bootstrap API
(with the default `region="auto"`) and exchanges the API key for a session token —
two sequential HTTPS round trips on the room-join critical path. Register the
plugin's `prewarm` function to move that work into process warm-up, before a job is
dispatched:

```python
from livekit import agents
from livekit.plugins import spatius


def prewarm(proc: agents.JobProcess):
    spatius.prewarm(proc)


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )
```

The spatius SDK caches the resolved region (5 minutes) and the prefetched session
token (until shortly before expiry) process-wide, so `AvatarSession.start()` in the
dispatched job reuses them and goes straight to the ingress WebSocket connect. It
also opens throwaway TLS connections to the console and ingress endpoints to prime
DNS and the shared TLS session cache. Warm-up is best-effort and never fails
process initialization; without it, everything resolves inline as before.

Session-token prefetch assumes the Spatius backend allows a token to back more than
one session; opt out with `spatius.prewarm(proc, prefetch_session_token=False)`.
