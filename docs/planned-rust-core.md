# Rust core architecture

This page extends [rust-core.md](rust-core.md).

## The run unit of work in Rust

The run unit of work and transport turn belong to `lk-agents-storage`.
Python and Node share its lease, ordering, and commit rules. Hosts run the
application handler, call models, execute tools, and report progress.

### Responsibilities

`lk-agents-storage` owns lane loading and commit, leases and renewal,
revision fences, transport task records, inbox policies, and turn ordering.
Application fields join the write set only through explicit context saves.
Router frames are application-persisted data; storage does not select agents
or reconstruct an execution stack.

### The host boundary

One trait, `TurnHost`, with one call:

```
handle(ctx: TurnContext) -> (Sink<Verdict>, Source<TurnEvent>)   // the a2a_session handler
```

The source yields a `checkpoint` before each model call and before the answer
streams, each answered with `continue` or `yield` through the sink, then the
`output`. Status frames and streamed tokens go from the handler to the wire
directly.

`TurnContext` carries the thread id, the lane, the message and the batch,
the history as JSON items, the state as JSON, the task id, the lane's task
rows, and a delegating caller's history when the message carried one. `HandlerOutput` is what the turn reads back: the output (text or a
typed value), the new items, the state, the usage, an `InputRequired`
question with its pending call, whether the handler yielded, and the
handler's error. Every value crosses as JSON
through the generated records.

The A2A door acks between the two calls: `submit` makes the task durable,
the door sends the `submitted` `Task`, then `run` takes it through its
policy. There is no submitted callback.

In Python, `_core.Turn(thread, lane, host, options)` takes any object with
the generated host callbacks; `turn.submit(message)` returns the task id and
`turn.run(task)` runs it.

### Host responsibilities

Python and Node expose the developer API, construct runtime application
context, execute tools and model calls, and report progress. Rust owns the
managed storage turn. Context loading and saving remain explicit host API
operations against the supplied scoped store.

## Storage bindings

Storage host records and methods are declared through `#[host(..)]` beside
their owning Rust items. Python and Node expose the same generated records
and binding surface. Requests and results cross as validated records.

## The layout check in CI

The layout check enforces the line cap (1,250 lines per `.rs` file under
`rust/crates`, generated output skipped), the rule that host records live
only in `lk-agents-abi`, and the rule that a core crate's `cargo tree` holds
no async runtime and no `lk-agents-runtime` ([Crates](#crates)). It runs in CI beside
`lk-agents-codegen --check`, so a stale generated `.pyi`, pydantic module or
`.d.ts` fails the build.

## Type enforcement

In CI, `docs/examples` and `examples/` are type-checked with mypy strict
(`tsc` for Node), and `assert_type` / `expectType` tests on the generics
(`Agent[Context, Output]`, `RunContext[Context]`,
`RunResult[Output]`, router frames).

## Cores and the runtime

The rule is in [rust-core.md](rust-core.md#cores-and-the-runtime): a crate
that decides is a core, `handle(event, now) -> effects`, and all I/O lives in
`lk-agents-runtime`. The voice driver, storage, worker, and inference follow this boundary.

### Event ownership

Each core handles one event at a time. Model streams, media, playback,
host API calls, room changes, and timers arrive as explicit inputs. The
runtime performs I/O from emitted effects and returns correlated result events.
Event order is observable, reproducible, and controlled in tests.

### Crates

Core crates contain deterministic decisions. Async I/O and provider adapters
live in `lk-agents-runtime`; bindings expose the language-specific surface.

| Crate | Responsibility |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `lk-agents-abi`       | the vocabulary: records, chat context, events and effects, the host-record table                                                              |
| `lk-agents-voice`     | the session core and the data-plane core                                                                                                       |
| `lk-agents-worker`    | `WorkerState`                                                                                                                                  |
| `lk-agents-storage` | what decides: the schema, statements, the run's lease and fence, the turn's policy |
| `lk-agents-inference` | what decides: reconnect, retry, fallback across providers |
| `lk-agents-media` | decode, encode, resampling |
| `lk-agents-runtime`   | the tokio runtime, the provider traits and streams, sockets and HTTP, the rtc proxies, the clocks, the voice host API and every core's event queue, the host bridge |

The bindings keep registration (`lib.rs`); runtime support lives in
`lk-agents-runtime` behind its `python` and `node` features, expanded in
place like every other item. Dependencies point down, and no core depends on
the runtime:

```
lk-agents-py, lk-agents-node                        the bindings: registration
lk-agents-runtime                                   all I/O, the one tokio crate
lk-agents-voice, -worker, -storage, -inference, -media   the cores
lk-agents-abi                                       the vocabulary
```

`lk-agents-fuzz` tests the cores with no runtime: it is the voice core's
event queue. `lk-agents-runtime`'s own tests may use one.

### The voice core

The session core owns generation planning, tool lifecycle decisions, IVR,
AMD, fillers, transcription timing, room state, and end-of-turn decisions.
Provider objects, sockets, outputs, and resource lifetimes belong to the runtime.

**The step.** `handle(event, now) -> effects` and `wake_at() -> Option<Ts>`.
Whoever owns the queue (the runtime, or the fuzzer in tests) calls
`handle(Event::Wake, now)` when `now` reaches `wake_at()`; a wake is due
before any input at the same instant. `now` is the session clock and the
wall time it maps to, so `ChatItem.created_at` comes from the step. Ids are
counters seeded once at construction; nothing in the core reads entropy.
Decisions run inside the step; the effects are only what leaves it. The
journal carries no time of its own: whoever reads it stamps what it observes.

**Resources.** Providers, outputs, inputs and recorders live in a table in
the runtime. An event names one by `ResourceId` and carries a record of what
the core needs to know about it (kind, sample rate, capabilities, label);
the core never holds the object.

**Events in**, each a typed record:

| Source | Events |
| -------------- | -------------------------------------------------------------------------------------------------------------- |
| the host | every API call with its call id and raw arguments, validated by the core; hook and tool results with theirs |
| hearing | VAD ticks, STT interims and finals, end-of-turn predictions, overlap verdicts, leg opened, failed or ended |
| models | an LLM's tokens, tool calls, end and failure; TTS audio produced, ended, failed; realtime server events |
| the data plane | playout started, progressed, finished, reset; the position a cut stopped at; frames forwarded per leg |
| the room | participant linked, unlinked, track changes, closed |
| time | `Wake` |


**Effects out**, each a typed record:

| Kind       | Effects                                                                                                   |
| ---------- | --------------------------------------------------------------------------------------------------------- |
| answers    | the result of an API call, by call id; a speech done, with its outcome                                     |
| host view  | the values the host reads synchronously: a handle's state, the resolved options, the linked participant    |
| host       | session events, hook calls, tool runs, each with an id                                                     |
| models     | open or cancel an LLM request; send text to TTS, end or cancel it; realtime sends                          |
| data plane | open, gate, pause, resume, flush or cut an output; enable or disable an input leg; ask for more audio     |
| journal    | the steps the rules read                                                                                  |

**Pending answers.** The core keeps one table per kind of request: API calls,
hook calls, tool runs, model requests. An entry is created with the effect
that asks and removed by the event that answers it or by the wake that times
it out. A session does not settle while any table is non-empty, so "every API
call returns" is a property of the core. Ids are never reused; a cancel
settles its entry, and an event for a settled id is dropped and journalled.

**The host's view.** The runtime keeps, per session and per handle, the values
the host reads synchronously, written only by host-view effects. The one
exception is `interrupt()`: it sets the handle's `interrupted` at once and
sends the event, because `interrupted` means "the host asked, or the core
interrupted it", two monotone facts with two writers that cannot conflict.
Whether the audio was cut is the speech's outcome, which only the core
reports. Every other write (`allow_interruptions`, options) changes the view
only when the core's effect arrives. A fuzz rule checks after each settle that
every view equals the core's state, `interrupted` being the core's value or
the host's request.

**Streaming and backpressure.** LLM tokens are events; sentence splitting is
pure code in generation planning; text goes to TTS as soon as a sentence is
ready, so nothing waits on credits before the first audio. The backpressure
is on audio: the runtime reads a TTS stream only when the data plane asks for
more, and the data plane asks while its output buffer has room. A stall, a
cancel or an interruption is a state the core is in, never a future dropped
somewhere.

**The data plane.** 20 ms frames do not go through the decision core. A second
core, stepped beside it, owns the microphone fan-out to the STT and VAD legs
and the realtime socket, the TTS-to-output buffer, the transcription
synchroniser (words paced against played audio), the recorder tap and the
pre-connect buffer. Its gates change only by the decision core's effects; it
reports counters (frames forwarded per leg, samples played) and answers a cut
with the position it stopped at, so a cut never waits on a scheduler hop. The
two cores talk only through the queue owner, which in tests is the fuzzer:
the hop between them is an ordering the fuzzer chooses like any other.

### Determinism

A core's output depends only on its events and their `now`:

- no order-dependent `HashMap` or `HashSet` iteration with a random hasher;
  use ordered maps or deterministic ordering for decisions;
- no clock read, no entropy, no thread-local state, nothing shared with
  another thread;
- the layout check rejects `RandomState`, `Instant::now`, `SystemTime::now`
  and `rand` in core crates.

### Threads and the runtime

One core per session, handled one event at a time, no locks. A session has no
thread of its own: `lk-agents-runtime` queues its events and steps its cores
on the process's tokio runtime, which also runs all I/O. A process runs a
handful of runtime threads for all its sessions. The journal leaves the core
as effects and the runtime decides what to keep, so a two-hour call does not
hold its whole journal in memory. Work per event is proportional to what the
event changes: a VAD tick, about 30 a second, does not rescan every subsystem.

`python/loop_time.rs` (`LoopRuntime`, the executor for Python's virtual-time
tests) provides a runtime mode that steps the cores on the asyncio loop.

### Bindings and features

`#[host]` items in a core crate are synchronous: records, enums, plain
functions. Every async binding (an awaitable, a host proxy, a callback) is
expanded in `lk-agents-runtime`. The `cargo tree` rule is checked with the
default features, and the `python` and `node` features of a core pull nothing
async. `lk-agents-runtime` has a `voice` module for the voice host API and its
handle objects, mirroring the voice core: home by subject inside the crate
that does I/O. The rtc data types the core reads (`RoomEvent`,
`ParticipantIdentity`, `VideoFrameRef`) live in `lk-agents-abi`; the rtc
traits and fake room live in `lk-agents-runtime`.

Python uses the Rust driver, so `api/host.rs` and
the handle objects are written once, against the new boundary.

### Testing

| Layer      | How                                                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| core       | the fuzzer is the event queue: no executor, no clock but the virtual one it passes in                                               |
| scheduling | partial-order sampling over the pending events (which source goes next, the hop between the two cores included), bursts delivered before the core settles, every same-instant order |
| host API   | the fuzzer sends raw API arguments, so validation is fuzzed with everything else                                                     |
| runtime    | small, each adapter against a fake socket under a deterministic executor                                                             |
| production | a session's event log, recorded and replayed offline step by step; a failing call becomes a regression test                          |

Partial-order sampling is the policy gosim measured at 2.6 times the bugs of
uniform scheduling, with uniform still best on some programs, so the policy
is drawn per seed. Reach alone cannot show what an ordering change gains, so
each step is also measured in distinct findings, keyed by rule and kind of
subject, on the same seeds.

### Machine composition

Small machines own their state and implement `handle(event) -> Effects`.
The driver routes events and concatenates effects. Machines communicate
through typed internal signals dispatched in order until settled. The fuzzer
reorders external events, not internal signals.

### Event log handling

Production event logs can contain conversation text. Retention, redaction,
and access follow the application's data policy. Replay uses recorded input
events and clock values without repeating external effects.
