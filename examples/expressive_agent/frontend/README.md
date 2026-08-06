# Expressive agent demo frontend

A single-page demo for the [expressive agent](../README.md): a mood-tinted
aura driven by the agent's `lk.expression` transcription attribute, an
Expressive/Flat toggle, and a voice picker. Both controls ride on the agent
dispatch metadata (see `../protocol.py`), so changing either reconnects with a
freshly dispatched agent.

No build step: `index.html` + `app.js` + `style.css`, with `livekit-client`
from a CDN. `server.py` mints tokens and serves the page.

## Run

Put LiveKit Cloud credentials for the project where `expressive_agent` is
deployed in `.env.local`:

```
LIVEKIT_URL=wss://<project>.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

Then, from the repository root:

```bash
uv run python examples/expressive_agent/frontend/server.py
```

and open <http://localhost:8080>.

The deployed agent must be dispatchable by name (`expressive_agent`); deploy it
with `lk agent deploy` from `..`, or run it locally against the same project
with `uv run python ../agent.py dev`.
