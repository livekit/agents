---
name: operating-livekit-agents
description: 'Deploys and operates a LiveKit agent in production: shipping a version to LiveKit Cloud and rolling it back, secrets and configuration, the worker process model and prewarming, safe async inside worker processes, provider timeouts and degradation, graceful shutdown, SDK upgrades, and observability. Use when the user says "deploy my agent", "roll back the deployment", "tail the agent logs", "the first call after a restart is slow", "attached to a different loop / event loop is closed", "prewarm the VAD", "shut down without dropping calls", "upgrade livekit-agents safely", "correlate logs by session", or is changing an agent codebase that is already live. Not for designing the agent (building-livekit-agents) or reproducing one bad conversation locally (debugging-livekit-agents).'
license: MIT
metadata:
  author: livekit
---

# Operating LiveKit agents

Everything after the agent works: getting a version onto LiveKit Cloud, keeping it fast and alive
under real load, and changing it without breaking what's already running. The commands live under
`lk agent`; read `lk agent --help` and each subcommand's help rather than trusting this skill for
flags — it deliberately doesn't restate them. `reading-livekit-docs` has the deployment and
observability docs.

## Deploying to LiveKit Cloud

The shape is stable even as the flags move:

1. **A project directory is bound to an agent** by a config file the CLI writes (`livekit.toml`).
   Commands run from that directory find the agent without an id.
2. **The agent ships as a container.** The CLI can generate a Dockerfile for the project, or you
   bring a prebuilt image. Run the container's start command locally (`lk agent start`) before the
   first deploy — it's production mode, with production logging and a shutdown drain, and it is
   not what `dev` mode runs.
3. **Each deploy creates a version** and rolls it out. `status`, `versions`, and `logs` (build logs
   and deploy logs are separate) tell you what's live and why a rollout failed.
4. **Secrets are injected as environment variables** and managed apart from the code — never
   baked into the image or committed. Changing secrets restarts the agent.
5. **Rollback returns to a previous version.** How instant that is depends on the plan; the docs
   say. Know the rollback command before you need it.

After a deploy, verify with the same tools you'd use on a stranger's agent: `status` for the
rollout, `logs` for the first minutes, and a real conversation — `running-livekit-simulations` can
run a scenario file against the deployed agent by name, which is the cheapest end-to-end check that
the thing serving traffic is the thing you meant to ship.

## The worker process model

Both SDKs run sessions in worker processes spawned from a parent. Misunderstanding this is the
single most common source of production-only bugs.

**The parent prewarms; children inherit.** Load expensive, read-only resources — VAD and turn
detection models, persistent clients — once in the parent through the SDK's prewarm hook, and
each session inherits them without re-loading. Everything shared this way must be read-only or
concurrency-safe; mutating parent state from a child is undefined behavior. Don't prewarm
session-specific state, and don't prewarm what costs more memory than it saves — every byte in the
parent is in every child's footprint. Then **verify the child actually uses the prewarmed
instance**: the classic mistake is prewarming a model and having session code load a fresh one
anyway, so the prewarm did nothing and startup is still slow.

**The framework owns the event loop.** Never create a new async runtime inside a worker, and never
block on an async call from a synchronous constructor to force a result. If initialization needs
async work, load lazily on first use from an already-async method, or split construction from an
awaited `initialize` step. When you see errors about events bound to a different loop, a loop
already running, tasks destroyed while pending, or a closed loop that can't be reused, the cause is
almost always one of those two things — trace back to where a runtime was created or a sync path
awaited something.

## Providers

STT, TTS, LLM, VAD and any backend will fail in production: rate limits, timeouts, overload,
outages. Set timeouts — a call that hangs is worse than one that fails fast. Distinguish transient
failures worth retrying from persistent ones that need a fallback or escalation. Degrade to a
meaningful spoken response, never to silence. Log provider response times, because rising latency
is usually the first sign of an outage.

Adding a provider to an existing agent: check whether the SDK already has a plugin before writing
one; follow how the codebase already initializes, configures, and handles errors for its other
providers; run its config through the same pipeline; and test under realistic latency, not just
happy-path responses.

## Performance: measure before you change anything

The metric users feel is **time to first audio** — from the caller finishing to the agent starting
to speak. Break it down before optimizing: connection, provider initialization, first inference,
tool execution. Watch **context growth** over a long call; unbounded history means every turn is
slower than the last.

The anti-patterns worth grepping for:

- **Synchronous I/O in an async context.** One blocking HTTP call or file read freezes audio for
  every concurrent operation in that process.
- **Loading during a call** — a model or connection established on first use inside a session is
  latency the caller hears. Move it to prewarm or session setup.
- **Unbounded context** with no summarization.
- **Every tool registered globally** regardless of conversation phase; each definition costs every
  LLM call.

**Endpointing and turn detection** are tuning parameters, not defaults to accept. Too aggressive and
the agent talks over people who pause to think; too conservative and every reply feels late. The
right setting depends on the use case — a support agent handling complex questions wants patience,
a quick-answer assistant wants speed. Measure with audio simulations (`running-livekit-simulations`
scores turn-taking and perceived latency) rather than by feel.

## Shutdown and upgrades

In production mode the server drains on a termination signal: it stops taking new jobs, finishes
active ones up to a drain timeout, then exits. Two things break this. An orchestrator grace period
shorter than the drain timeout kills calls mid-sentence — match them. And post-call work (writing
state, publishing events, finalizing recordings) that isn't tied to the job's lifecycle gets cut
off — make sure it completes inside the drain. Dev mode has no drain, so shutdown behavior has to
be tested in start mode.

The SDK moves fast and breaks things across versions. Pin the version. Read the changelog with
`reading-livekit-docs` before upgrading. Upgrade on a branch, run the test suite, then have real
conversations and run the simulation suite — some regressions only appear in actual dialogue. Watch
latency and error rates for the first hours after the deploy.

## Observability

Three things make a production incident tractable instead of a mystery:

- **Every log line carries the session id**, so one call can be traced end to end.
- **Phase boundaries are timed** — connected, first user speech, first agent response — so you
  can see *where* latency entered, not just that it did.
- **Provider metrics are tracked separately** from application logs, so a degrading provider is
  visible without digging.

LiveKit Cloud provides agent logs, session insights, and tracing you can export to your own
observability provider; the docs describe what's available and how to wire it.

## Debugging a production failure

Classify first, then reproduce:

- **Loop and task errors** — the worker process model, above. Find where a runtime was created or
  a sync path awaited.
- **Provider timeouts vs. degradation** — a single timeout is transient; a rising latency trend is
  systemic. Provider response-time logs tell you which.
- **Silent tool failures** — a tool that raised was caught by the framework and the user heard a
  vague reply or nothing. Tools need explicit error handling that returns something the model can
  say.
- **Connection instability** — understand the SDK's built-in reconnection before adding retries
  on top of it; log connection state changes.

Then take it local: reproduce the conversation with `debugging-livekit-agents`, pin the cause as a
test with `testing-livekit-agents`, and if it was a whole-conversation failure, add a scenario so it
can't come back quietly.

## Changing a codebase that's already live

Read the project's own docs and conventions first — they override anything general. Read two or
three modules like the one you're about to add before writing it, and follow how they're
initialized, configured, tested, and wired into the config pipeline. Use the test fixtures and mock
patterns that already exist rather than building parallel ones; a second way to do something that
already has a way confuses every future contributor. Mock at the boundary — the provider call, the
database — and exercise the real module logic.

## Related skills

- Designing the agent in the first place: `building-livekit-agents`
- Reproducing a bad conversation locally: `debugging-livekit-agents`
- Pinning a production bug as a test: `testing-livekit-agents`
- End-to-end checks against a deployed agent: `running-livekit-simulations`
- Deployment and observability docs, changelogs: `reading-livekit-docs`
