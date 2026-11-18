# Web chat

A text chat agent and a one-page web app for it, kept in [threads](../../docs/threads.md):
each browser chats with the agent and sees only its own conversations.

| File      | What                                                                           |
| --------- | ------------------------------------------------------------------------------ |
| `main.py` | The agent and its `chat` text door.                                            |
| `web.py`  | The page's backend: serves `web/`, mints tokens and reads the threads.         |
| `web/`    | The page: `index.html`, `style.css` and `app.js`. No build step, no framework. |

## How it works

- **A scope per browser.** The page keeps a random id in `localStorage`; its threads live in
  the scope `browser:<id>`. The id is the browser's only credential: a real app ties the scope
  to its signed-in user instead.
- **Sending is A2A.** `web.py` mints a token with `Credentials.thread_grant_token()`: to start
  a thread in the browser's scope, or to continue one of its threads, once it checked the thread
  is the browser's. The page posts `SendStreamingMessage` with it to the agent's `chat` endpoint
  on LiveKit Cloud and appends the answer's chunks as they arrive. The first message has no
  `contextId`, so the platform starts a thread and returns its id.
- **Reading is the threads API.** `web.py` lists the scope's threads and reads a thread's items
  with the project key (`storage.threads.list_threads()`, `Thread.list_items()`), and the page
  renders them, also after each reply: the thread is the source of truth, the page keeps no copy.

## Run

With the environment of [`../README.md`](../README.md), from this directory, run the agent and
the page side by side:

```sh
lk agent dev
uv run web.py
```

Then open http://localhost:8000.
