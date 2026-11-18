# Web chat

A chat agent with a one-page web app. Each browser keeps its own conversations, stored in
LiveKit threads.

- `agent.py`: the agent. It streams its answers and is reached through LiveKit Cloud.
- `web.py`: the page and its backend. It holds the project key, mints the browser's tokens
  and reads its threads.
- `web/`: the page itself. No build step, no framework.
- The browser sends messages straight to the agent through LiveKit Cloud, and reads history
  through `web.py`.

## Run

```sh
export LIVEKIT_AGENT_DEPLOYMENT=dev
lk agent dev agent.py    # terminal 1: the agent
uv run web.py            # terminal 2: the page, on http://localhost:8000
```
