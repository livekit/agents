# Rust core: planned

Not built yet. This extends [rust-core.md](rust-core.md).

## The run unit of work in Rust

The run unit of work and the turn belong to `lk-agents-storage`, not to
Python: a Node package gets the same mechanism, and its tests are the
crate's, run once against a real agent-db. Python and Node keep the
developer surface and the three things only a host can do: run the handler
(call the model, execute a tool) and report progress.

### What moves

From `storage/run.py` and `storage/policies.py`:

| Python                                                                      | Rust                                                                                                                                                           |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LaneHandleImpl.run`, `_acquire`, `_open_concurrent`, `_commit`, `_abandon` | `run::Lane::open(RunOptions) -> Run`; `Run::commit`, `Run::abandon`; the renewal timer                                                                         |
| `Run`: history, state, agent stack, events, tasks, lease, fence             | `run::Run`: the loaded rows, the write set, the fence; history items and state cross as JSON                                                                   |
| `Tasks`: `write`, `submit`, `wait`, `pending_ask`, `by_message_id`          | `run::Tasks`                                                                                                                                                   |
| `TurnRunner.run_turn` and the policies                                      | `turn::Turn::submit(message) -> task id`, which writes the submitted row, and `turn::Turn::run(task) -> TurnResult`: one state machine, the policy a parameter |
| `_submit`, `_pickup`, `_fail_unrun`                                         | the same names on `turn`                                                                                                                                       |
| `_turn`, `_finish`                                                          | the turn step; the host runs the handler inside it                                                                                                             |
| `Message`, `channel_of`, `resolve_policy`, `resolve_debounce`               | `turn::Message` and the same functions                                                                                                                         |

The turn follows the lane rules of [lanes.md](lanes.md) and
[storage.md](storage.md): runs execute in arrival order, a waiter on an
`interrupt` or `queue` endpoint fails `lease-timeout` after the fixed 60 s
cap, a turn whose lease holder died
fails `worker-lost` and is never re-run, and a failed save ends the task
failed with `livekit/reason=storage-unavailable`.

### The host boundary

One trait, `TurnHost`, with three calls:

```
handle(ctx: TurnContext) -> HandlerOutput   // the a2a_session handler, or Agent.run()
on_update(task_id, text)                    // a working status frame to the wire
on_token(task_id, delta)                    // streamed answer text
```

`TurnContext` carries the thread id, the lane, the message and the batch,
the history as JSON items, the state as JSON, the task id, the lane's task
rows, a delegating caller's history when the message carried one, and the
`Boundary` the handler polls before each model call and before the answer
streams. `HandlerOutput` is what the turn reads back: the output (text or a
typed value), the new items, the state, the usage, an `InputRequired`
question with its pending call, whether the handler yielded, and the
handler's error. Every value crosses as JSON
through the generated records.

The A2A door acks between the two calls: `submit` makes the task durable,
the door sends the `submitted` `Task`, then `run` takes it through its
policy. There is no submitted callback.

In Python, `_core.Turn(thread, lane, host, options)` takes any object with
the three coroutine methods; `turn.submit(message)` returns the task id and
`turn.run(task)` runs it.

### What stays in Python

`protocol.py` (the types a developer reads), `agentdb.py` (the binding over
`_core`), `live.py`, `voice/agent_run.py`, `resolve.py` (thread resolution
is request handling), and the A2A door, which calls `_core.Turn.submit` and
`_core.Turn.run` instead of `TurnRunner`. `rows.py` is generated.

### Order

| Step | Content                                                                                                                                                                           | State                                                                   |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 1    | The lease TTL and the renewal interval as `StorageOptions` fields (default 10 s and 3 s); the Python tests on a real agent-db                                                     | done                                                                    |
| 2    | `run::Lane`, `Run`, `Tasks` in Rust; `run.py` reduced to the binding                                                                                                              | the crate side is built; `run.py` still runs its own unit of work       |
| 3    | `turn` in Rust with `TurnHost`, exposed as `_core.Thread`, `_core.Turn`, `_core.Boundary`; `policies.py` reduced to `Message` and the context type; the A2A door on `_core.Turn` | the crate side and `_core` are built; the door still calls `TurnRunner` |
| 4    | The storage cases as crate tests (`run_local.rs`, `turn_local.rs`)                                                                                                                | done                                                                    |

Each step leaves the demos running.

## The storage binding through `#[host]`

`lk-agents-py/src/storage.rs` and `turn.rs` are hand-written pyo3 over
`lk-agents-storage`. They move to `#[host(..)]` declarations on the storage
crate's own items, expanded in place like every other core crate, so
`lk-agents-py` is back to `lib.rs` and `runtime.rs` and Node gets the same
surface from the same declarations. Requests, results and rows cross as
generated records, not dicts.

## The layout check in CI

The layout check enforces the line cap (1,250 lines per `.rs` file under
`rust/crates`, generated output skipped), the rule that host records live
only in `lk-agents-abi`, and the rule that a core crate's `cargo tree` holds
no async runtime and no `lk-agents-runtime` ([Crates](#crates)). It is ported to this repository and runs in CI, beside
`lk-agents-codegen --check`, so a stale generated `.pyi`, pydantic module or
`.d.ts` fails the build.

## Type enforcement

In CI, `docs/examples` and `examples/` are type-checked with mypy strict
(`tsc` for Node), and `assert_type` / `expectType` tests on the generics
(`Agent[Deps, State, Output]`, `RunContext[Deps, State]`,
`RunResult[Output]`, `AgentTask[Output, TaskState]`).

## Cores and the runtime

The rule is in [rust-core.md](rust-core.md#cores-and-the-runtime): a crate
that decides is a core, `handle(event, now) -> effects`, and all I/O lives in
`lk-agents-runtime`. This section is the design for the crates that do not
follow it yet, the voice driver first.

### Why

A voice session is many independent sources producing events over time: the
person through STT, VAD and end-of-turn, several models streaming, outputs
reporting playback, the room, the developer's API calls, timers. Its bugs are
ordering bugs. The fuzz found a refused speech that still changed the agent's
state, microphone frames that stopped reaching the recogniser, API calls
whose futures were dropped, and speeches marked done without the host being
told. With state shared by concurrent tasks, every `.await` is a point where
other code changes that state, and the interleavings are invisible in the
code. With one owner handling one event at a time, every race is an order of
events: explicit, enumerable and replayable.

### Crates

The cores keep their names and lose every async dependency. All I/O moves
into one crate, `lk-agents-runtime`, the way everything Python lives in
`lk-agents-py`.

| Crate                 | Holds                                                                                                                                          | From today                                                              |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `lk-agents-abi`       | the vocabulary: records, chat context, events and effects, the host-record table                                                              | `lk-agents-abi` without its provider traits, spawner and clocks          |
| `lk-agents-voice`     | the session core and the data-plane core                                                                                                       | `lk-agents-voice` without its tasks, pumps and host API                 |
| `lk-agents-worker`    | `WorkerState`                                                                                                                                  | `lk-agents-worker` without its transports                               |
| `lk-agents-storage` | what decides: the schema, statements, the run's lease and fence, the turn's policy | `lk-agents-storage` without its connection and awaited scripts |
| `lk-agents-inference` | what decides: reconnect, retry, fallback across providers | `lk-agents-inference` without its `Native*` clients, `transport.rs`, MCP and HTTP calls |
| `lk-agents-media` | decode, encode, resampling | unchanged, but it reaches tokio through `lk-agents-abi` until abi is pure |
| `lk-agents-runtime`   | the tokio runtime, the provider traits and streams, sockets and HTTP, the rtc proxies, the clocks, the voice host API and every core's event queue, the host bridge | all of the above it leaves, `lk-agents-rtc`, and `runtime.rs` from the bindings |

The bindings keep registration (`lib.rs`); their `runtime.rs` moves into
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

The driver has part of the shape: one loop owns the session (`Driver::run`),
takes one `Input` at a time with a merge rule ("a due timer fires before any
same-instant input"), and asks every subsystem for its next deadline. Hooks
cross as an effect and an answer (`HostEvent::Hook`, then
`Command::HookDone`), a leg opens the same way (`ProviderCall::OpenLeg`, then
`Input::LegOpened`), and realtime already has `send` and `next_event`. What
breaks the model:

- the tasks the loop spawns decide on their own: the generation pipeline with
  its tools, IVR, AMD, fillers, the transcription synchroniser, keyterms,
  room I/O and the end-of-turn leg (about 40 spawn sites in 19 files);
- events are not data: `Command` carries provider and output objects
  (`Arc<dyn Stt>`, `Arc<dyn AudioOutput>`) and answers on `oneshot`
  channels (46 sites);
- today's `Effect` is not the boundary: `execute` performs `Decision`s,
  branches on state and reads the clock, and `Trace::push` reads it too;
- the host reads shared state synchronously: `SpeechHandle` is built from
  `watch`, `Mutex` and atomics, and `publish_gate` reads flags the host
  writes from its own thread.

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

| Source         | Events                                                                                                         | Today                                                   |
| -------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| the host       | every API call with its call id and raw arguments, validated by the core; hook and tool results with theirs   | `Command`, `Input::Host`, `api/host.rs`                 |
| hearing        | VAD ticks, STT interims and finals, end-of-turn predictions, overlap verdicts, leg opened, failed or ended      | `Input::{Stt, Vad, Eot, Overlap, LegOpened, LegEnded}`  |
| models         | an LLM's tokens, tool calls, end and failure; TTS audio produced, ended, failed; realtime server events        | inside the generation tasks; `Input::Realtime`          |
| the data plane | playout started, progressed, finished, reset; the position a cut stopped at; frames forwarded per leg          | `GenEvent::{FirstFrame, PlaybackStarted}`, sink reports |
| the room       | participant linked, unlinked, track changes, closed                                                            | `Input::{Room, RoomIo}`                                 |
| time           | `Wake`                                                                                                         | `Wake::Timer`                                           |

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

- no `HashMap` or `HashSet` iteration with a random hasher (IVR's loop
  detector sums floats in `HashMap` order today, so its threshold flips with
  the hasher); ordered maps or a fixed hasher;
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
tests) becomes a runtime mode that steps the cores on the asyncio loop.

### Bindings and features

`#[host]` items in a core crate are synchronous: records, enums, plain
functions. Every async binding (an awaitable, a host proxy, a callback) is
expanded in `lk-agents-runtime`. The `cargo tree` rule is checked with the
default features, and the `python` and `node` features of a core pull nothing
async. `lk-agents-runtime` has a `voice` module for the voice host API and its
handle objects, mirroring the voice core: home by subject inside the crate
that does I/O. The rtc data types the core reads (`RoomEvent`,
`ParticipantIdentity`, `VideoFrameRef`) move to `lk-agents-abi`; the rtc
traits and the fake room move to `lk-agents-runtime`.

Python switches to the Rust driver after the migration, so `api/host.rs` and
the handle objects are written once, against the new boundary.

### Testing

| Layer      | How                                                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| core       | the fuzzer is the event queue: no executor, no clock but the virtual one it passes in                                               |
| scheduling | partial-order sampling over the pending events (which source goes next, the hop between the two cores included), bursts delivered before the core settles, every same-instant order |
| host API   | the fuzzer sends raw API arguments, so validation is fuzzed with everything else                                                     |
| runtime    | small, each adapter against a fake socket under a deterministic executor                                                             |
| production | a session's event log, recorded and replayed offline step by step; a failing call becomes a saved case                               |

Partial-order sampling is the policy gosim measured at 2.6 times the bugs of
uniform scheduling, with uniform still best on some programs, so the policy
is drawn per seed. Reach alone cannot show what an ordering change gains, so
each step is also measured in distinct findings, keyed by rule and kind of
subject, on the same seeds.

### Migration order

Each step keeps the fuzz gate green and lowers neither the survey's reach nor
its distinct findings. The saved corpus is re-recorded when the boundary
changes.

Scheduler control comes first: after the boundary's resources by id, steps 1
to 3 remove the spawned tasks, then the host view and validation finish step
0, then steps 4 and 5. The data plane is one core per session; inference's
fallback across providers is its own core in `lk-agents-inference`, so the
voice core sees one model. Python switches to the Rust driver as its own
project after step 5.

Each step rewrites its area as small machines, each owning its state and
answering `handle(event) -> Effects`; the driver only routes events and
concatenates effects, with one effect type. Python is the behaviour reference
through the fuzz rules, not the shape of the code; comments keep a Python name
only where the behaviour would surprise without it.

0. The boundary: `handle(event, now)` and `wake_at()` replace `Driver::run`'s
   loop, which becomes the runtime's queue owner; resources by id, call ids on
   every API call, answers by id instead of `oneshot`; the host view;
   validation in the core; outward effects, with decisions run inside the step;
   no clock read on the loop's path; ids from counters. From here the
   fuzzer owns the queue, so partial-order sampling and bursts work over every
   source, the still-spawned tasks' reports included.
1. The data plane: frame forwarding, output buffering, the synchroniser, the
   recorder tap, pre-connect audio and the realtime audio push. The gate is
   already published once per cycle (`publish_gate`).
2. Generation with its tools: the LLM stream, sentence splitting, TTS, tool
   runs and playout become core state driven by model events, with audio
   demand for backpressure. Tools cannot go first: they are dispatched inside
   the generation task, and their results decide its next step.
3. The other spawners: IVR, AMD, fillers, keyterms, room I/O, the end-of-turn
   leg.
4. The crates: `lk-agents-runtime` is created with the provider traits, the
   spawner, the clocks, the async utilities (`tts/stream_adapter`,
   `utils/{stream,duplex,bounded_lossy,monotone}`, `turn/{detector,interruption}`,
   `evals`, `worker/job`) and `lk-agents-rtc`; then the voice and worker I/O
   move into it. The layout check's `cargo tree` rule turns on per core crate
   as it becomes pure.
5. Storage and inference: only what decides becomes a core (the run's lease
   and fence and the turn's policy; reconnect, retry and fallback); their I/O
   scripts move to `lk-agents-runtime` as they are. `WorkerState`'s `tick`
   becomes `Event::Wake` and `wake_at()`.

### Open questions

- How long a production event log is kept and what it may contain: it holds
  the conversation's text.
