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
lk agent dev agent.py    # terminal 1: the agent
uv run web.py            # terminal 2: the page, on http://localhost:8000
```

## Reach another agent

`web.py` builds the address `/agents/<agent>/<deployment>/<endpoint>` from environment variables:

| Variable | Default | Set it to |
| --- | --- | --- |
| `LIVEKIT_AGENT_NAME` | `web-chat` | the agent's name |
| `LIVEKIT_AGENT_ENDPOINT` | `chat` | the `a2a_session(endpoint=)` to message |
| `LIVEKIT_AGENT_DEPLOYMENT` | `_` | a named deployment such as `dev`, the same value the worker has |

An agent in LiveKit Cloud's `production` deployment, and a worker started without a
deployment, is served at `_`: leave the variable unset. `production` itself is not an address
and is read as `_`.
