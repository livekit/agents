# Web chat

A text chat agent and a one-page web app for it, kept in [threads](../../docs/threads.md):
each browser chats with the agent and sees only its own conversations.

| File         | What                                                                           |
| ------------ | ------------------------------------------------------------------------------ |
| `agent.py`   | The agent and its `chat` text door.                                            |
| `backend.py` | The app's backend: mints the page's tokens and reads its threads.              |
| `static.py`  | Serves the page, `web/`.                                                       |
| `web/`       | The page: `index.html`, `style.css` and `app.js`. No build step, no framework. |

## How the pieces connect

```
browser ──── page ─────────────────────▶ static.py (localhost:8000)
   │
   ├──── tokens, history ──────────────▶ backend.py (your backend, localhost:8001)
   │                                        │ reads threads with the project key
   │                                        ▼
   └──── messages (A2A) ──▶ LiveKit Cloud ──▶ agent.py (the agent, wherever it runs)
                               threads live here
```

- `backend.py` is your app's backend. It holds the project key, so the browser never sees it.
- The agent is never called directly: the browser sends to its endpoint on LiveKit Cloud,
  `https://<project>.livekit.cloud/agents/web-chat/_/chat/v1`, and Cloud routes the message to
  the worker, here a process on your machine under `lk agent dev`, once deployed one on Cloud.
- Cloud keeps the threads; both sides read and write them there.

## How it works

- **A scope per browser.** The page keeps a random id in `localStorage`; its threads live in
  the scope `browser:<id>`. The id is the browser's only credential: a real app ties the scope
  to its signed-in user instead.
- **Sending is A2A.** `backend.py` mints a token with `Credentials.thread_grant_token()`: to start
  a thread in the browser's scope, or to continue one of its threads, once it checked the thread
  is the browser's. The page posts `SendStreamingMessage` with it to the agent's `chat` endpoint
  on LiveKit Cloud and appends the answer's chunks as they arrive. The first message has no
  `contextId`, so the platform starts a thread and returns its id.
- **Reading is the threads API.** `backend.py` lists the scope's threads and reads a thread's items
  with the project key (`storage.threads.list_threads()`, `Thread.list_items()`), and the page
  renders them, also after each reply: the thread is the source of truth, the page keeps no copy.

## Run

With the environment of [`../README.md`](../README.md), from this directory, in three terminals:

```sh
export LIVEKIT_AGENT_DEPLOYMENT=dev   # in each: the agent registers under it, the page calls it
lk agent dev agent.py                 # the agent: registers with LiveKit Cloud
uv run backend.py                     # the backend, on http://localhost:8001
uv run static.py                      # the page, on http://localhost:8000
```

Then open http://localhost:8000.
