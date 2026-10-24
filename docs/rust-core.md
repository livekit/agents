# Rust core

The framework's core is one Rust workspace, `rust/`, shared by the Python
and Node packages. Behaviour that both languages need (the worker's
connection to LiveKit, the thread storage client, the voice driver, the
inference clients, media) is written once in Rust and reached from each
language through a generated binding. The packages keep what only a host
language can do and what a developer reads, writes or subclasses.

Planned work on the core is in [planned-rust-core.md](planned-rust-core.md).

## What lives where

| Rust, `rust/crates`                                                                                                                                                   | Python, `livekit-agents`                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| the worker control plane: registration, load and status, job assignment, the WebSocket and WebTransport transports, the endpoint streams cloud opens per HTTP request | process management and IPC (`ipc/`): the process pool, job processes, signals, the ASGI app the endpoint streams are served into |
| the agent-db client: the `wsdb` protocol, the thread schema, every SQL statement, the lease and its renewal, the answer streams, the threads service client         | `storage/`: the developer-facing types, thread resolution, the A2A door                                                         |
| the voice driver, the `Native*` inference clients and fallback adapters, MCP, media decode and encode, resampling                                                     | the developer surface: `Agent`, `AgentSession`, `LiveSession`, tools, tasks, `RunContext`, plugins                               |

Plugins stay Python. A Python provider reaches the core through a host
proxy: the core calls the Python object behind a Rust trait (see "Host
callbacks").

The Python package drives the worker and storage through `_core`. The voice,
inference, media and rtc crates are compiled into `_core` and register their
classes; the Python voice pipeline does not call them.

## Crates

| Crate                   | Subject                                                                                                                                                                                                                                                                                                                            |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lk-agents-abi`         | The provider seam (`Stt`, `Tts`, `Llm`, `Vad`, `Realtime` traits) and the data vocabulary every other crate is written against, one module per subject named like the Python package. The host records (`#[derive(Host)]`). Behind its `python` feature: the registration table, the fork guard, the host bridge. Performs no I/O. |
| `lk-agents-bindgen`     | The expansion of `#[host(..)]`: an item's pyo3 binding, emitted in the crate that owns the item. Shares its attribute parser with `lk-agents-codegen`.                                                                                                                                                                             |
| `lk-agents-host-derive` | The proc macros: `#[derive(Host)]`, `#[host(..)]`, `Vocabulary`. Reached through `lk-agents-abi`.                                                                                                                                                                                                                                  |
| `lk-agents-codegen`     | The generator binary: reads every declaration and writes the `.pyi` stubs, the pydantic modules, the TypeScript interfaces and the napi records. Also `check-layout`, the architecture check on the voice crate's module boundaries.                                                                                               |
| `lk-agents-media`       | Container decode to PCM, ogg/opus encode, image encode, resampling. Pure Rust by default; `blingfire`, `images`, `ffmpeg` and `resample` add the C and C++ parts.                                                                                                                                                                  |
| `lk-agents-rtc`         | RTC capabilities supplied by the host SDK: rooms, participants, tracks, streams and audio sources as host proxies, plus a fake room for tests.                                                                                                                                                                                     |
| `lk-agents-inference`   | The `Native*` inference-gateway clients (STT, TTS, LLM, realtime, VAD, end of turn, AMD, interruption), the fallback adapters, the MCP client. The socket appears only in `transport` and `native`.                                                                                                                                |
| `lk-agents-voice`       | The voice session driver. Directories mirror the Python voice modules (`agent_session/`, `agent_activity/`, `generation/`, `audio_recognition/`, `room_io/`); `trace.rs` is the `Step` journal.                                                                                                                                    |
| `lk-agents-worker`      | The worker control plane: `Worker`, `WorkerHost`, the sans-IO `WorkerState`, the transports, `EndpointStream`.                                                                                                                                                                                                                     |
| `lk-agents-storage`     | The thread storage client: `wsdb`, `schema`, `statements`, `run`, `turn`, `streams`, `grpc`, `threads`, `token`. See [storage.md](storage.md).                                                                                                                                                                                           |
| `lk-agents-fuzz`        | The voice behavioural suite (see "Testing").                                                                                                                                                                                                                                                                                       |
| `lk-agents-py`          | `livekit.agents._core`, the pyo3 extension. A shell.                                                                                                                                                                                                                                                                               |
| `lk-agents-node`        | `@livekit/agents-core`, the napi-rs binding.                                                                                                                                                                                                                                                                                       |

Dependencies point down:

```
lk-agents-py, lk-agents-node                    the bindings
lk-agents-fuzz                                  tests the voice driver
lk-agents-voice      -> rtc, media, abi
lk-agents-inference  -> media, abi
lk-agents-worker, lk-agents-rtc, lk-agents-media -> abi
lk-agents-storage    -> host-derive
lk-agents-abi        -> host-derive -> bindgen
lk-agents-codegen    -> bindgen                 (a build tool, not linked)
```

`lk-agents-abi` is the bottom of the product crates: it cannot see voice or
inference, so nothing that needs them lives there. `lk-agents-storage` is
host-neutral and depends on no other product crate. The workspace builds
from the sibling `livekit/rust-sdks` checkout for `livekit-protocol` (the
`livekit.agent` protobuf package and the worker wire) and `livekit-token`.

## Boundaries

### RTC

The core never links the `livekit` crate or `livekit-ffi`. Rooms,
participants, tracks and audio sources belong to the host's SDK and reach
Rust as host proxies (`lk-agents-rtc`); no rtc type appears in a core
signature. User-defined audio and text I/O goes through `voice/io`.

### Host callbacks

The core calls into Python in three shapes, all generated:

| Shape              | Declared as                                                               | Used by                                                                        |
| ------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| provider proxy     | a bare `#[host(proxy(host = "livekit.agents.llm.LLM"))]` on the abi trait | a Python STT, TTS, LLM, VAD or realtime model driven by the core               |
| `open(kind, args)` | one bridge for every kind, returning an awaitable                         | `WorkerHost`: `job_request`, `launch_job`, `terminate_job`, `load`, `endpoint` |
| host trait         | a Rust trait a Python object implements                                   | `TurnHost`: `handle`, `on_update`, `on_token`                                  |

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
  (`lk-agents-bindgen`). Only `lk-agents-py` turns the `python` features on.
- `lk-agents-codegen` parses the same source with `syn` for what a proc
  macro cannot see (order, doc comments) and writes the `.pyi` stubs and
  pydantic modules under `livekit-agents/livekit/agents/_generated/`, the
  TypeScript interfaces, and the napi records under
  `lk-agents-node/src/generated/`. Generated files carry a first-line mark.

```bash
cd rust
cargo run -p lk-agents-codegen -- --write   # regenerate in place
cargo run -p lk-agents-codegen -- --check   # exit 1 when a generated file would change
```

`lk-agents-py` is `lib.rs` (registration) and `runtime.rs` (`init`, the
fork guard re-export). The hand-written exceptions are
`lk-agents-py/src/storage.rs` and `lk-agents-py/src/turn.rs` (see the bindings rules).

## Testing

| Suite               | What it proves                                           | Run                                 |
| ------------------- | -------------------------------------------------------- | ----------------------------------- |
| crate unit tests    | each crate's rules, with a negative case per rule        | `cargo test`                        |
| `lk-agents-fuzz`    | the voice driver against behavioural contracts           | `cargo test -p lk-agents-fuzz`      |
| storage crate tests | statements, the run and the turn against a real agent-db | `cargo test -p lk-agents-storage`   |
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
the case's policy, saved with the case, chooses between the next pending event
and a turn for one of the driver's ready tasks: arrival order drains the queue
first, as the channel does; partial-order sampling picks events by random
priority within each source's order and interleaves task turns at random. The
search (`src/search/`) plays the person, the developer's program
and the providers, each acting on what it observed, with a per-seed dial of
uniform moves from none to almost all. `rules/` holds one guarantee per file,
each with its own examples; `replay/` holds saved cases. `cargo test` runs
the rules' examples, the determinism check and the saved corpus; the search
campaign is behind `--ignored` (`tests/survey.rs`, `tests/soak.rs`). The
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
`WorkerState` (`lk-agents-worker/src/state.rs`) is the model. The crate
layout, the voice design and the migration are in
[planned-rust-core.md](planned-rust-core.md#cores-and-the-runtime).

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
  same name (`AgentSession`, `AgentActivity`, `SpeechHandle`,
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
- **`lk-agents-py` and `lk-agents-node` are shells:** `lib.rs`
  (registration) and `runtime.rs` only.
- **Host records are declared only in `lk-agents-abi`.** Enforced by the
  layout check.
- **Extend the generator, never hand-write binding code.** A Rust
  declaration may not be bigger than the Python it replaces.
- **Known exceptions:** `lk-agents-py/src/storage.rs` and
  `lk-agents-py/src/turn.rs` are hand-written until both move to `#[host]` on
  `lk-agents-storage`.

### Typed end to end

The framework is typed end to end, in Python and in Node.

- **No `Any` in the public API.** Every public function, class and
  callback is fully typed. The generated `.pyi` and `.d.ts` are the typed
  surface of the core; the `.d.ts` carries no `any`.
- **Generics carry types:** `Agent[Deps, State, Output]`,
  `RunContext[Deps, State]`, `RunResult[Output]`, `AgentTask[Output, TaskState]`.
- **Typed data at boundaries.** Persisted state, deps, output and tool
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
