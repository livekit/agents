# Console

The frontend of the Agents 2.0 v0 demos: a chat with a front desk
(plan 10.1 demo 2) with the session's database beside it, live (demo 3). One
Next.js app in the Blender visual language of memdata; `DESIGN.md` says how
it is drawn.

## What it talks to

- The front desk's A2A endpoint on the node, directly from the browser, with a
  session grant: `message/stream` over fetch and SSE, `tasks/resubscribe` to resume a reply after a reload.
- The console API (`console_api.py`, beside this README) for everything else: the
  grant (`GET /api/token?sub=`), the subject's threads (`GET`/`POST
  /api/sessions`), one thread's database pushed every second over SSE
  (`GET /api/sessions/{id}/watch`), its history and background tasks from
  the threads service (`GET /api/sessions/{id}/history`, `/tasks`), and the stream relay
  (`GET /api/streams/{id}`).

## Run it

The stack of `examples/README.md` must be up: livekit-server on 7880,
agent-db on 7780, 7781, 7783 and 7793, an example's two agents. Then the console API,
from this directory with the workers' environment, signed for the node's key so
the token it mints is the node's:

```sh
uv run python console_api.py                                   # :7795
```

The console lists every `examples/*/console.toml`, top right. It boots on the first one
running; pick another to talk to it, and one that is not running shows the command that
starts it and connects by itself once it is up. The dev server's origin is allowed.

```sh
cd console
pnpm install
pnpm dev                 # http://localhost:3000
```

`NEXT_PUBLIC_AGENTDB_DEBUG_URL` points the app at the console API
(`.env.example`; default `http://localhost:7795`).

## Use it

Type to the front desk; Enter sends, Shift+Enter breaks the line, Escape
clears the draft, Cmd+K starts a new conversation. The reply streams; the
expert's portal steps show as progress lines; a tool that needs the person
shows its question. Reload and the conversation comes back from
the thread's history; reload while a reply streams and it resumes through
`tasks/resubscribe`; a saved session that no longer exists is dropped and the
page starts a new conversation. Every turn says which task, which end state
and which worker and pid served it; the task id opens that task in the
inspector.

The speech button in the composer (the waveform) starts a call into the
same conversation: the room token comes from `app/api/token` with the
session id in the dispatch metadata, so the voice agent joins the session on
screen, and nothing connects before the click. The call bar over the
composer mutes the microphone, picks another one when the browser has
several (components-react's device list, the one agents-ui's track control
uses; the choice is kept for the next call) and ends the call. Text during a
call lands in the same session.

The demo's knobs are in the Demo view at the foot of the rail. The policy sets
`livekit/channel`: under interrupt (a2a) a burst ends the first task
`canceled` with `livekit/interrupted` and the second answers both; under inbox
(sms) one reply answers both and the first task carries `livekit/mergedInto`.
The composer shows the policy as one small label when it is off its default.

The rail on the left opens one view beside the conversation: the subject's
conversations (named by the console's own `app/api/title` after the first
turn, text or voice; a session nothing was said in is not listed), then the
session's database: the lanes with their lease counting down, the task
table, the journal, the inbox, and a task's stream, replayable. The panel's
width and the open view persist.

## Check it

```sh
pnpm check               # typecheck, lint, the token guard, unit tests
pnpm test:e2e            # Playwright on chromium and webkit, against the live stack
```

The e2e suite holds real conversations (three turns with the expert, a reload
mid-reply, an inbox burst, the lease changing hands between the two workers)
and writes screenshots to `/tmp/lk-console` (`E2E_SHOTS` to change it). It
compares states, metadata and the panels, never the model's wording.
