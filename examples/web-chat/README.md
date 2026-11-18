# Web chat

A text chat agent and a one-page web app for it, kept in [threads](../../docs/threads.md):
each browser chats with the agent and sees only its own conversations.

| File      | What                                                                           |
| --------- | ------------------------------------------------------------------------------ |
| `main.py` | The agent and its `chat` text door.                                            |
| `web.py`  | The page's backend: serves `web/` and mints the browser's tokens.              |
| `web/`    | The page: `index.html`, `style.css` and `app.js`. No build step, no framework. |

## How it works

- **Tokens per browser.** The page keeps a random id in `localStorage`. `web.py` mints it a
  grant to list and start the threads of its own scope, and, for each thread, a grant to read
  and continue that one thread once it checked the thread is the browser's. Both come from
  `Credentials.thread_grant_token()` and last 15 minutes. The id is the browser's only
  credential: a real app ties the scope to its signed-in user instead.
- **Sending is A2A.** The page posts `SendStreamingMessage` to the agent's `chat` endpoint on
  LiveKit Cloud and appends the answer's chunks as they arrive. The first message has no
  `contextId`, so the platform starts a thread in the browser's scope and returns its id.
- **Reading is the threads API.** The sidebar is `ListThreads` for the scope, and a chat is
  rendered from `GetThreadItems`, also after each reply: the thread is the source of truth, the
  page keeps no copy. The browser reads both from `https://api.livekit.cloud`.

## Run

With the environment of [`../README.md`](../README.md), from this directory, run the agent and
the page side by side:

```sh
lk agent dev
uv run web.py
```

Then open http://localhost:8000.
