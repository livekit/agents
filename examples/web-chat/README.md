# Web chat

A text chat agent and its one-page web app, served from the agent's own endpoint and kept in
[threads](../../docs/threads.md): anyone who opens the page can chat, and each browser sees
only its own conversations.

| File         | What                                                                                   |
| ------------ | -------------------------------------------------------------------------------------- |
| `main.py`    | The agent, its `chat` text door, and the public routes: the page and its token.       |
| `static/`    | The page: `index.html`, `style.css` and `app.js`. No build step, no framework.         |

## How it works

- **A grant per browser.** The page keeps a random id in `localStorage` and asks `/token` for a
  thread grant scoped to it (`list`, `create`, `read`, `write`, 15 minutes), minted with
  `Credentials.thread_grant_token()`; it asks again once the grant expires. The id is the
  browser's only credential: a real app scopes the grant to its signed-in user instead.
- **Sending is A2A.** The page posts `SendStreamingMessage` to `chat/v1`, next to itself, with
  the grant as its bearer token, and appends the answer's chunks to the reply as they arrive.
  The first message has no `contextId`, so the platform creates a thread in the browser's
  scope and returns its id; later messages send it.
- **Reading is the threads API.** The sidebar is `ListThreads` for the scope, and a chat is
  rendered from `GetThreadItems`, also after each reply: the thread is the source of truth,
  the page keeps no copy. The browser reads both directly from LiveKit's public API,
  `https://api.livekit.cloud/v0/projects/p_/threads`, with the same grant.
- **The page is routes.** `server.http.get(..., public=True)` serves `static/index.html` at
  `/agents/web-chat/<deployment>/` and the rest of `static/` under it, beside the `chat` door
  and `/token`.

## Run

With the environment of [`../README.md`](../README.md), from this directory:

```sh
lk agent dev
```

Then open `https://<project>.livekit.cloud/agents/web-chat/_/` (`.staging.livekit.cloud` on
staging, where the threads API is `https://api.staging.livekit.cloud`). The browser calls the
threads API cross-origin, so that API must answer CORS for the page's origin.
