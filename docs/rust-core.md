# Rust core

The framework's core is one Rust workspace, `rust/`, shared by the Python
and Node packages. Behaviour that both languages need (the worker's
connection to LiveKit, the thread storage client, the voice driver, the
inference clients, media) is written once in Rust and reached from each
language through a generated binding. The packages keep what only a host
language can do and what a developer reads, writes or subclasses.

## What lives where

| Rust, `rust/crates`                                                                                                                                                   | Python, `livekit-agents`                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| the worker control plane: registration, load and status, job assignment, the WebSocket and WebTransport transports, the endpoint streams cloud opens per HTTP request | process management and IPC (`ipc/`): the process pool, job processes, signals, the ASGI app the endpoint streams are served into |
| the agent-db client: the `wsdb` protocol, the thread schema, every SQL statement, the lease and its renewal, the answer streams, the threads service client         | `storage/`: the developer-facing types, thread resolution, the A2A door                                                         |
| the voice driver, the `Native*` inference clients and fallback adapters, MCP, media decode and encode, resampling                                                     | the developer surface: `Agent`, `TextSession`, `VoiceSession`, `VoiceTask`, tools, `RunContext`, plugins                               |

Plugins stay Python. A Python provider reaches the core through a host
proxy: the core calls the Python object behind a Rust trait (see "Host
callbacks").

Python and Node reach the Rust runtime and its cores through generated
bindings. The runtime drives voice, worker, storage, inference, and media
operations; language hosts expose application callbacks and provider plugins.

## Crates

| Crate | Responsibility |
| --- | --- |
| `lk-agents-abi` | Data vocabulary: typed records, chat items, events/effects, resource references, and host records; no I/O or async runtime |
| `lk-agents-bindgen` | Expands binding declarations in their owning crate; shares its parser with codegen |
| `lk-agents-host-derive` | The `Host`, `host`, and `Vocabulary` proc macros |
| `lk-agents-codegen` | Generates Python stubs/models/provider bases, TypeScript interfaces, napi records, and architecture checks |
| `lk-agents-media` | Decode, encode, image conversion, and resampling; optional codec features add C/C++ implementations |
| `lk-agents-inference` | Deterministic reconnect, retry, and provider-fallback decisions |
| `lk-agents-voice` | Session decision core and data-plane core; generation, tools, IVR, AMD, transcription timing, and turn decisions |
| `lk-agents-worker` | Worker control-plane decisions, including `WorkerState` |
| `lk-agents-storage` | Schema, statement plans, lease/fence decisions, and run commit rules |
| `lk-agents-a2a` | Text endpoint policies and transport-turn decisions over storage |
| `lk-agents-runtime` | All async I/O: provider traits/clients, connections, clocks, RTC proxies/fakes, resource tables, event queues, and async host APIs |
| `lk-agents-fuzz` | Drives cores directly through typed events and behavioral contracts |
| `lk-agents-py`, `lk-agents-node` | Language registration shells for the generated API |

Dependencies point down:

```
lk-agents-py, lk-agents-node                          registration
lk-agents-runtime                                   I/O and queue ownership
lk-agents-voice, -worker, -storage, -inference, -media decision/data-plane cores
lk-agents-a2a -> lk-agents-storage                    transport policy decisions
lk-agents-abi                                       shared data vocabulary
```

No core depends on `lk-agents-runtime` or an async executor. Provider and RTC
objects stay in runtime-owned tables; cores receive typed resource descriptions
and IDs. Shared RTC data records live in ABI; RTC traits, proxies, and the fake
room belong to the runtime. `lk-agents-fuzz` depends on the cores without the
runtime; runtime-adapter tests can use a deterministic executor.

Proc macros and codegen are build tools, not runtime dependencies on I/O.
The workspace reuses `livekit-protocol`, `livekit-api`, and `livekit-token`
from LiveKit's Rust SDK rather than duplicating their wire definitions.

## Boundaries

### RTC

The core never links the `livekit` crate or `livekit-ffi`. Rooms,
participants, tracks and audio sources belong to the host's SDK and reach
Rust as runtime-owned host proxies; core signatures contain only ABI data
records and resource IDs, never live RTC objects. User-defined audio and text I/O goes through `voice/io`.

### Host callbacks

The runtime calls into Python in three shapes, all generated:

| Shape              | Declared as                                                               | Used by                                                                        |
| ------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| provider proxy     | a bare `#[host(proxy(host = "livekit.agents.llm.LLM"))]` on the runtime provider trait | a Python STT, TTS, LLM, VAD or realtime model driven by the core               |
| `open(kind, args)` | one bridge for every kind, returning an awaitable                         | `WorkerHost`: `job_request`, `launch_job`, `terminate_job`, `load`, `endpoint` |
| stream proxy       | `#[host(proxy)]` on a trait whose method returns a sink and a source     | `TurnHost`: `handle` returns the turn's events, checkpoints answered by verdicts |

A call is scheduled on the event loop that built the Rust object; its result
comes back on a channel; dropping the Rust future cancels the Python task.
The GIL is held to build a call and to read its result, never across an
await.

### The `_core` module

`livekit.agents._core` is built by maturin from `rust/crates/lk-agents-py`
(`livekit-agents/pyproject.toml`, `[tool.maturin]`) and ships inside the
`livekit-agents` wheel, built `cp310-abi3`: one build serves Python 3.10
and later. Every class, function
and exception on it is declared beside its Rust item and registers itself
through a link-time table; the `#[pymodule]` body drains the table.
`lk-agents-py/src/core_names.json` freezes the public names and a test
compares `dir(_core)` against it. `_core.pyi` names what the Python package
imports; the generated classes' stub is `_generated/_core_classes.pyi`.

### Fork safety and the runtime

Importing `_core` starts nothing: no runtime, thread or socket. The
forkserver imports the plugin set and forks, and `fork()` clones only the
calling thread, so a runtime started at import would hang the child.

The process runs one tokio runtime, created on first use (after the fork)
and never dropped. `_core.init(role)` declares the process; a process that
never calls it is a job. The fork guard is a pid check at use: every entry
point calls `check_pid()` first, and a runtime used in a forked child that
did not re-init raises `RuntimeError` instead of deadlocking, until the
child calls `init` itself. No `pthread_atfork` handler is installed, so
fork followed by exec stays allowed.

### Logging

The core logs through the `log` facade with structured keys:

```rust
log::warn!(target: "livekit.agents", lane = lane.as_str(); "the lease was lost");
```

`pyo3-log` forwards every record into Python's `logging`, under the
`livekit.agents` logger, with the keys as `extra`. A log line is a side
effect; nothing reads it as data.

## Bindings

A core item is exposed where it is declared:

```rust
#[derive(Host)]
#[python(pydantic, module = "events")]
#[node(interface)]
#[binding(emit)]
pub struct AgentStateChanged { .. }
```

- `#[derive(Host)]` declares a host record. Under the declaring crate's
  `python` feature it expands to the record's pyo3 conversions or class.
- `#[host(..)]` on a function, `impl`, trait, stream or error expands in
  place to its pyo3 function, methods, proxy, stream class or exception
  (`lk-agents-bindgen`). Async items and host proxies are declared in the
  runtime; synchronous core items and ABI data remain free of an executor.
- `lk-agents-codegen` parses the same source with `syn` for what a proc
  macro cannot see (order, doc comments) and writes the `.pyi` stubs and
  pydantic modules under `livekit-agents/livekit/agents/_generated/`, the
  TypeScript interfaces, and the napi records under
  `lk-agents-node/src/generated/`. Generated files carry a first-line mark.
- A `#[host(proxy)]` trait with `#[python(provider(..))]` is also the Python
  base class plugins subclass (`stt.STT`, `tts.TTS`, `llm.LLM`, `vad.VAD`,
  `llm.RealtimeModel` and their streams, in `livekit-agents-next`): each
  method's `#[python(..)]` role gives its hook and its `_core_<method>`, so
  a trait method the class does not answer fails the codegen.

```bash
cd rust
cargo run -p lk-agents-codegen -- --write   # regenerate in place
cargo run -p lk-agents-codegen -- --check   # exit 1 when a generated file would change
```

The language bindings are registration shells. Runtime initialization, the
fork guard, async host bridging, and voice handles live in
`lk-agents-runtime` behind its language features and are generated from their
own declarations.

## Managed turns and the host boundary

The storage and A2A cores decide lease ownership, fencing, arrival order,
inbox policies, task transitions, and commit plans. Runtime adapters execute
the database requests, renewals, streams, and transport writes. A storage
script can await ordered I/O, but stateful policy decisions remain in cores.
Python and Node use the same turn machinery.

The internal host opens a handler as an event source and verdict sink:

```text
handle(TurnContext) -> (Sink<Verdict>, Source<TurnEvent>)
```

The host emits checkpoints before model calls and answer delivery. The turn
answers each with `continue` or `yield`, then consumes the handler's final
output. `TurnContext` carries thread/lane identity, incoming messages, history,
explicitly saved fields, transport task records, and permitted caller history.
The handler output carries new history items, typed output or an input-required
outcome, usage, explicitly staged field writes, and failure/yield information.
Cross-boundary records are generated from ABI declarations.

Submission makes the task record durable before acknowledging it; execution
then applies the lane policy. Lease loss fences writes; worker loss does not
implicitly rerun application code. Storage failures surface as failed transport
outcomes under the [storage contract](storage.md#the-unit-of-work).

Application context construction and field load/save stay explicit in the host.
Only staged fields join the managed write set. `TextSessionState` is saved
application data, not a stack reconstructed by storage. The internal checkpoint
protocol is distinct from the public [semantic run stream](agents.md#semantic-events).

## Voice core execution

### Steps, time, and resources

The session core exposes `handle(event, now) -> effects` and
`wake_at() -> Option<Ts>`. Its queue owner delivers a due wake before another
input at the same instant. `now` supplies the session clock and its wall-time
mapping; item timestamps derive from that step. IDs are counters seeded once
at construction. The core reads neither clocks nor entropy.

Providers, room objects, inputs, outputs, and recorders live in a runtime
resource table. Events identify them by `ResourceId` and carry their required
data: kind, capabilities, sample rate, and label. The core never retains a
provider object, channel, callback closure, or independently running future.

### Inputs and effects

| Source | Inputs |
| --- | --- |
| Host | Validatable API arguments, hook answers, and tool results, all with request IDs |
| Hearing | STT, VAD, end-of-turn and overlap reports; leg lifecycle events |
| Models | LLM text/tool events, TTS output and completion, realtime events |
| Data plane | Playout progress, cuts, gates, and forwarded-frame counters |
| Room | Participant and track lifecycle events |
| Time | A wake with supplied clock values |

Effects request provider I/O, host tools/hooks, resource changes, media gates,
output operations, host-view updates, and journal observations. The runtime
executes them and reports answers as correlated events. Decisions occur in
the step, not in effect execution.

### Pending answers and synchronous host views

Maintain pending entries for API calls, hooks, tools, and model requests.
Each entry is settled by its answer, cancellation, or timeout wake. Settled IDs
are never reused; late answers are ignored and journalled. End-of-operation
cleanup cannot declare success while it still owes an answer.

The runtime maintains the values host code reads synchronously, such as speech
state, resolved options, and participant identity, from host-view effects.
`interrupt()` is the narrow exception: its host-request flag is monotone and
can be recorded immediately while the core separately reports the actual
speech outcome. Other option changes become visible through the core's
accepted update; tests compare host views with settled core state.

### Streaming and the data plane

LLM tokens are input events. Pure generation logic splits sentences and requests
TTS as soon as text is ready. Audio demand provides backpressure: runtime
adapters pull provider audio while the output buffer has room. Cancellation,
stalls, and interruptions are explicit state transitions.

A separate data-plane core handles 20 ms media frames without routing every
frame through the decision queue. It owns microphone fan-out, conversion,
output buffering, transcription timing against played audio, recording taps,
and pre-connect buffering. Decision-core effects change its gates; it reports
playout positions and counters through the queue owner. A cut reports where
playback stopped. The boundary between the two cores is visible to the fuzzer.

### Scheduling, composition, and replay

A session owns no dedicated thread. The process runtime steps many sessions,
one event per core at a time. Work per event follows what changed; a VAD tick
must not scan every subsystem. Smaller machines own their state and exchange
typed internal signals in a defined order until settled; the driver routes
signals and concatenates effects. External events, not internal signals, are
reordered by the scheduling harness.

Core tests need no executor. They cover raw host-argument validation, pending
answers, same-instant ordering, bursts, resource lifecycle, and consistency
between cores and host views. Runtime adapters use fake sockets and a
deterministic executor. Sampling policies and seeds are retained with cases;
compare distinct findings as well as reach under the same inputs.

The runtime exports journal observations without retaining an unbounded call
log in memory. Production event logs can contain conversation text; retention,
redaction, and access follow application policy. Offline replay supplies the
recorded events and clock values to cores without repeating external effects.

## Testing

| Suite               | What it proves                                           | Run                                 |
| ------------------- | -------------------------------------------------------- | ----------------------------------- |
| crate unit tests    | each crate's rules, with a negative case per rule        | `cargo test`                        |
| `lk-agents-fuzz`    | the voice driver against behavioural contracts           | `cargo test -p lk-agents-fuzz`      |
| storage crate tests | the statements, the run and the turn against a real agent-db | `cargo test -p lk-agents-storage`   |
| binding tests       | the `_core` surface (`core_names.json`), host bridging   | `cargo nextest run -p lk-agents-py` |
| Python tests        | the package over the built `_core`                       | `uv run pytest --unit`              |

**Parity contract.** The voice driver journals every decision as a `Step`
(`lk-agents-voice/src/trace.rs`). The journal is the contract: a rule reads
the journal, never internal state.

**Voice fuzz.** `lk-agents-fuzz` runs a real voice session against dumb
fakes: every event at a provider boundary (a transcript, a detector tick, a
playout report, a socket drop) is an input the case sends. Runs are
deterministic: paused virtual time, session-scoped ids. The run owns the
driver's queue: it calls `handle` itself and delivers a due wake first. Then
the case's saved scheduling policy chooses which external event is delivered
next. Partial-order sampling preserves each source's causal order, varies
inter-source ordering, and delivers bursts before settlement. The hop between
the decision and data-plane cores is another schedulable external boundary;
internal machine signals retain their defined order. The
search (`src/search/`) plays the person, the developer's program
and the providers, each acting on what it observed, with a per-seed dial of
uniform moves from none to almost all. `rules/` holds one guarantee per file,
each with its own examples. `cargo test` runs the rules' examples, the
determinism check and the regression tests; the search campaign is behind
`--ignored` (`tests/survey.rs`, `tests/soak.rs`). A seed alone reproduces its
run: `VOICE_SEED=<n>` reruns one, and a fixed finding becomes a hand-written
regression test. The
contracts are in `rust/crates/lk-agents-voice/TEST_STRATEGY.md`; what the
search reaches and misses is in `rust/crates/lk-agents-fuzz/COVERAGE.md`.

**Storage.** `lk-agents-storage/tests/{agentdb_local,run_local,turn_local}.rs`
run against agentstore through the public API of a staging project
(`LIVEKIT_URL`, with `LIVEKIT_API_KEY` and
`LIVEKIT_API_SECRET`), each test on a fresh database. The host of a turn
is a scripted fake: a handler that returns, raises, asks, or sleeps until
told. The Python storage and A2A tests use the same project through
`tests/agentdb_fixture.py`. Both skip when no project is set.

## Build and test

The build needs a Rust toolchain and a checkout of `livekit/rust-sdks`
beside this repository (the path dependencies in `rust/Cargo.toml`).

```bash
make install                  # uv sync: builds livekit-agents, _core included, with maturin
```

The editable install rebuilds `_core` when a Rust source changes
(`[tool.uv] cache-keys` in `livekit-agents/pyproject.toml`).

From `rust/`:

```bash
cargo build                         # the host-neutral crates, no pyo3
cargo test                          # their tests
cargo build -p lk-agents-node       # the node addon, on its own
cargo nextest run -p lk-agents-py   # the python binding tests (links libpython)
cargo run -p lk-agents-codegen --bin check-layout   # the voice architecture check
```

The two bindings are workspace members but not default members. Cargo
unifies features across one invocation, and `lk-agents-py` turns on the
`python` feature of the core crates, which links libpython into every
artifact built with it. Build each host in its own invocation; never build
the node addon in the same command as `lk-agents-py` or `--workspace`.

## Codebase rules

These hold for every change under `rust/`.

### Boundaries

- **The core never touches rtc.** No `livekit` crate, no `livekit-ffi`.
  The boundary is `io.py` and the rtc proxy.
- **Rust never does IPC or process management.** `ipc/` stays Python.
  Rust runs inside a job, plus the worker control plane.
- **Fork-safe import.** The native module is inert at import. One tokio
  runtime per process, started after the fork and never dropped. The fork
  guard is a pid check at use: a runtime used in a forked child without
  re-init fails loudly. Fork followed by exec stays allowed; no
  `pthread_atfork`.
- **One wheel.** `livekit.agents._core` ships inside the
  `livekit-agents` wheel, built `cp310-abi3` (one build for 3.10 and later).
  No separate package, no ABI version check.

### Cores and the runtime

What decides is a core; what talks is `lk-agents-runtime`, one crate for all
I/O, the way everything Python lives in `lk-agents-py`. A script that only
awaits I/O in sequence (a statement, then the next) stays a script in the
runtime; the decisions inside it (a lease, a policy, a retry) are a core.
`WorkerState` (`lk-agents-worker/src/state.rs`) is the model.

- **A core is a state machine.** `handle(event, now) -> effects`, and
  `wake_at() -> Option<Ts>` for when it next wants `Event::Wake`: no
  `async`, no spawn, no lock, no timer of its own. `now` carries the session
  clock and the wall time.
- **A core is deterministic.** Its effects depend only on its events and
  their `now`: no clock read, no entropy (ids are counters), no iteration
  over a randomly hashed map, nothing shared with another thread. The layout
  check rejects `RandomState`, `Instant::now`, `SystemTime::now` and `rand`
  in core crates.
- **Only `lk-agents-runtime` depends on an async runtime.** No other
  product crate has tokio, futures or an executor in its dependency tree
  with default features, and no core depends on `lk-agents-runtime`.
  `#[host]` items in a core are synchronous; every async binding expands in
  the runtime. Enforced by the layout check over `cargo tree`.
- **Events and effects are data.** Typed records, never closures or
  channels. A core's event log replays it exactly.
- **Every request is answered.** An API call, a hook call, a tool call and
  a provider request each carry an id; the answer is an event or effect with
  that id, a timeout is a wake, and a core does not settle while an answer
  is owed. An id is never reused; an answer for a settled id is dropped and
  journalled.
- **The host reads a view.** What the host reads synchronously (a speech
  handle's state, resolved options) is a view the runtime keeps, written only
  by the core's effects. The host's own monotone requests (`interrupt()`)
  show in the view at once; every other write waits for the core.
- **The runtime decides nothing.** `lk-agents-runtime` owns the tokio
  runtime, sockets, streams, the real clock, the rtc proxies and the event
  queues; it executes effects and turns what happens into events. A branch
  on state belongs in a core.
- **Tokio is for I/O.** A client that sends a request and returns the
  answer is a runtime function; a client with state (reconnects, retries, a
  fallback across providers, a stream protocol) keeps that state in a core.
- **Frames bypass the decision core.** Audio and video frames move through
  a data plane, itself a core: its gates change only by the decision core's
  effects, and it reports counters back as events.
- **One owner, one event at a time.** A core belongs to one sequence and
  handles its events in order, never two at once. The runtime owns the
  queue and calls `handle`; in tests the fuzzer owns it. No thread per
  session: cores are stepped on the shared runtime.
- **Tests drive the core directly.** A test feeds events and reads effects;
  the fuzzer is the core's event queue, and its scheduling policy chooses
  which pending event goes next. A deterministic executor is only for
  `lk-agents-runtime`'s own tests.

### Names and layout

- **Names follow the Python reference.** The same concept carries the
  same name (`VoiceSession`, `AgentActivity`, `SpeechHandle`,
  `schedule_speech`), snake_cased. The structure may differ.
- **1,250 lines per `.rs` file** under `rust/crates`. Generated output
  (first-line mark) is skipped. Enforced by the layout check.
- **No confetti files.** A hard file is fixed by order and signposting
  first: a contents list in the module docs, sections in the order the
  subject goes through them, each helper beside its user. Split only into a
  handful of files whose names a newcomer would guess.
- **Home by subject, not by dependency shape.** A module lives with what
  it knows about. Voice directories mirror the Python modules.

### Bindings

- **Bindings are generated in place.** `#[host(..)]` on the core item
  expands to pyo3 and napi behind the `python` and `node` features. One
  declaration gives pyo3, napi, `.pyi` and `.d.ts`.
- **`lk-agents-py` and `lk-agents-node` are registration shells.** Runtime
  support and async bridging belong to `lk-agents-runtime`.
- **Host records are declared only in `lk-agents-abi`.** Enforced by the
  layout check.
- **Extend the generator, never hand-write binding code.** A Rust
  declaration may not be bigger than the Python it replaces.

### Typed end to end

The framework is typed end to end, in Python and in Node.

- **No `Any` in the public API.** Every public function, class and
  callback is fully typed. The generated `.pyi` and `.d.ts` are the typed
  surface of the core; the `.d.ts` carries no `any`.
- **Generics carry types:** `Agent[Context, Output]`,
  `RunContext[Context]`, `RunResult[Output]`, session frames.
- **Typed data at boundaries.** Persisted fields, output and tool
  arguments are validated types, never raw dicts in the developer surface.
- **Typed wire.** A2A frames, grants, threads service calls and extension
  metadata are typed records generated from one declaration (proto or
  `#[host]`), shared by Python and Node.
- **Enforcement.** The generated `.pyi` and `.d.ts` must match the Rust
  declarations (`lk-agents-codegen --check`); drift fails the build. The
  generics have `assert_type` / `expectType` tests.

### Code

- **Lean: delete over add.** Extend the existing mechanism before
  adding one. Delete dead code and stale docs. Inline single-use helpers.
  Comments are one line of why; no narration.
- **A method earns its place.** It keeps an invariant (writes several
  fields together, emits effects or answers, validates) or names a
  non-trivial computation used twice. Otherwise it is a `pub(crate)` field:
  a private field with a getter is for state only an invariant-keeping
  method may write, and there are no bare setters.
- **Split by what the developer sees.** A struct that bundles concerns is
  split by the outcome its users see, named after the Python API
  (`callbacks.rs`, `handoff.rs`, `drain.rs`, `dtmf.rs`), never after internal
  mechanics.
- **Shared crates.** Never vendor or re-implement LiveKit's own crates
  (`livekit-protocol`, `livekit-api`, `livekit-token`); a missing field gets
  an upstream PR. A third-party dependency is a case-by-case cost call
  (build time, wheel size).
- **Logging.** `log::info!(target: "livekit.agents", key = value; "msg")`,
  forwarded by `pyo3-log` into Python `logging` with the keys as `extra`. A
  log line is a side effect, never data passed across the FFI.

### Tests

- **No test-only APIs.** A test observes through surfaces that exist
  (the public `_core` and Python API, journal steps the rules already read),
  or it is dropped.
- **Tests that can fail.** Every rule has a negative case. No asserts on
  a value's shape or a default. Duplicates are merged. Golden traces and the
  journal are the parity contract.
- **Fuzz fixes go in the harness.** An unreached variant is fixed in the
  generator or harness, after proving it from the driver's refusal. Never
  patch a product crate to please the harness. Thresholds come from the
  driver, not magic numbers.
