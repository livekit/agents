# AGENTS.md

## Build and Development Commands

This project uses **uv** as the package manager. All commands run from the repository root.

### Installation
```bash
make install          # Install all dependencies with dev extras (uv sync --all-extras --dev)
make build            # Rebuild the Python extension in place (after install)
make build-debug      # Rebuild with native debug symbols
make -C rust build     # Build every workspace crate, including FFmpeg
make rust-check       # Check every Rust crate and test target, including FFmpeg
```

For parallel worktrees, keep a private Cargo target directory per worktree.
Make uses `mbx` by default for Rust commands; install it with
`cargo install --locked mbx`. Use `CARGO_RUNNER=cargo` for ordinary Cargo.
See `docs/internals/development.md#local-builds` for setup and other cache tools.
Do not point concurrent worktrees at one writable `CARGO_TARGET_DIR`.

### Versioning

- Versions change only in release PRs (`livekit-agents@X.Y.Z`). Other PRs never edit `version.py` files or the `livekit-agents` version floor in plugin `pyproject.toml` files.
- Use a `patch` version bump by default.
- Do not use a `minor` or `major` version bump unless a human explicitly confirms the bump level.

### Code Quality
```bash
make format           # Format code with ruff
make lint             # Run ruff linter
make lint-fix         # Run ruff linter and auto-fix issues
make type-check       # Run mypy type checker (strict mode)
make check            # Run all checks (format-check, lint, type-check)
```

Rust commands live in `rust/Makefile`:
```bash
make -C rust clippy        # Clippy with the purity, I/O and base policies, denying warnings
make -C rust check         # Workspace Clippy, boundaries and codegen
make -C rust test-checks   # rust/check and codegen tests
```

### Testing
```bash
uv run pytest --unit                    # Run all unit tests
uv run pytest python/tests/test_tools.py --unit # Run a single unit test file
make unit-tests                         # Run unit tests that don't require cloud accounts
make test                               # Run the default hermetic suite
make bench                              # Run the standalone IPC benchmark
make bench-startup                      # Time start, dev and console launches to ready
make bench-host                         # Time the Rust-to-Python host boundary
```

#### Test categories

Every test module declares exactly one category via a module-level marker, and
each category has a matching `--<category>` selection flag. Selection happens
*before* test-module import, so unselected test modules are not imported.
Shared `conftest.py` fixtures and their dependencies still load. Categories are
registered in `pyproject.toml`; `python/tests/conftest.py` reads their literal markers.
Put test paths before category flags, whose optional argument is a provider name.

| Marker | Flag | Meaning |
|--------|------|---------|
| `pytest.mark.unit` | `--unit` | fast, hermetic, no external providers/credentials/network |
| `pytest.mark.audio_eot` | `--audio_eot` | hermetic audio end-of-turn / turn-detection suite |
| `pytest.mark.plugin("name")` | `--plugin [name]` | provider integration test (needs that provider's deps/keys) |
| `pytest.mark.stt("name")` | `--stt [name]` | speech-to-text suite, optionally for one provider; requires provider dependencies and credentials |
| `pytest.mark.realtime("name")` | `--realtime [name]` | realtime-model test |
| `pytest.mark.evals` | `--evals` | behavioral evals against the LiveKit inference gateway |
| `pytest.mark.docs` | `--docs` | tests for the docs-build tooling under `.github/` |
| `pytest.mark.e2e` | `--e2e` | real process and transport journeys; selected runs require live services |

```bash
uv run pytest --unit --audio_eot        # the CI unit gate (no cloud accounts)
uv run pytest --plugin openai           # only the openai provider tests
uv run pytest --list-categories         # list every module grouped by category, then exit
```

**Adding a test:** give the new module a category marker (`pytestmark =
pytest.mark.unit`, etc.) — collection fails with a hint if it lacks one. Run
pytest with the `--allow-uncategorized` option to temporarily disable this rule
(CI keeps it on by default).

### Running Agents
```bash
lk agent dev               # Development mode with hot reload (connects to LiveKit)
lk agent console           # Talk to the agent in the terminal
lk agent start             # Production mode
```
`lk agent` (the Go livekit-cli) runs `python main.py start [--dev]` or `python main.py console`;
those subcommands are its internal interface, not a user entry point.

### Linking Local python-rtc (for SDK development)
```bash
make link-rtc         # Link to local python-rtc with downloaded FFI artifacts
make link-rtc-local   # Build and link local rust SDK from source (requires cargo)
make unlink-rtc       # Restore PyPI version
make status           # Show current linking status
make doctor           # Check development environment health
```

## Architecture Overview

This is the `v2.0` branch: Agents 2.0, in beta. `docs/README.md` indexes the design, one feature
per page. Feature documents describe the end state without implementation-status labels;
inspect the implementation to determine what remains to change.

The feature documents define the intended API. Do not preserve conflicting behavior merely
because current code or tests rely on it. Discuss unresolved API choices before changing the
specification; implementation cleanup must follow the agreed contracts.

### Core Concepts
- **Fast brain / slow brain**: the pattern every example follows. A frontend agent talks to the
  person (persona, pacing, voice) and its one tool is a `RemoteAgent(...).as_tool()` to a backend agent that
  holds the tools, the state and the systems of record.
- **Agent** (`voice/__init__.py`): shared configuration, `Agent[Context, Output]`, and validated
  `output_type`, authored as a constructed instance with registered callbacks. `await agent.run(message, message_history=, context=)` executes that agent.
  Static instructions plus `@agent.instructions` functions; identity is an optional `name=`,
  else the module-level variable name. The loop itself is `AgentRun` in `lk-agents-core`.
- **Streaming**: `agent.run_stream(...)` and backend `session.run_stream(...)` are
  async context managers with one text/event consumer and an awaitable `result()`.
  They share `run()` semantics; cancellation and cleanup settle before final retrieval.
  Application context/session saves remain explicit. See `docs/agents.md`.
- **RunContext[Context]**: tools and instruction functions receive `ctx.context`,
  `ctx.thread`, `ctx.live_session`, `ctx.store`, and `ctx.update()`.
- **RunResult[Output]**: a discriminated result; `status="completed"` exposes the typed
  `output`, while `status="input_required"` exposes `question`. Tools can request input
  with `InputRequired(...)`; framework outcomes remain separate from `output_type`.
- **TextSession**: request-scoped text conversation orchestration, reconstructed from
  shared agent configuration and explicitly supplied saved state or a store binding.
  Each child requires `output_type`; accepted final output fills the parent tool call.
  Returns `result: RunResult` and a detached `state: SessionState`. `context=store.bind(app)`
  opts into Rust-owned incremental persistence; plain context execution reads snapshots without
  binding or saving them. See `docs/text-session.md`.
- **VoiceTask[Output]**: fast-brain/frontend live interactions in an active voice session.
  `.run()` resolves the session from invocation-local runtime context; callers do not
  pass it explicitly. Calling outside a live interaction fails before execution.
  Temporarily takes the floor, returns a typed result, and restores the parent without
  calling the parent's on_exit/on_enter hooks. Its
  parent await remains in memory. Not for stateless workloads, backend orchestration,
  background execution, or recovery across requests/deployments. See `docs/voice-tasks.md`.
- **Context persistence**: `persisted(...)` marks fields; `load_context` and `save_context`
  are explicit. An optional persisted SessionState field is hydrated by load_context and keeps
  its identity while explicitly bound. Session views, binding policy and checkpoints live in
  Rust with generated Python APIs. Runtime services remain application-supplied objects.
- **Thread**: one conversation (`AT_...` id, the A2A `contextId`), with persistence partitions,
  histories and transport tasks stored in its database (agent-db, or local SQLite).
- **Rust core** (`rust/crates/`, see `docs/internals/architecture.md`): the sans-I/O crates
  (`lk-agents-core` with its `a2a` and MCP logic, `-voice`, `-media`) are synchronous (events and
  time in, effects out); media is CPU work with no async runtime. `lk-agents-runtime` is only the
  I/O layer of voice and text: the voice executor, room I/O, avatar transport, recording
  ingress, AgentRun/TextSession drivers, the thin A2A door (executes core's storage effects by
  calling storage), thin MCP I/O, HostInvoker and the process runtime; it makes no decisions.
  Everything that can be faked at its boundary is ordinary async Rust, tested on its own and
  faked in the DST: `lk-agents-net` (the shared HTTP/SSE/WebSocket transport, the host stream
  adapter, the delegation A2A client, and telemetry's export: the OTLP exporter and the Cloud
  session report upload),
  `lk-agents-inference` (providers, retry/fallback/reconnect, the TTS audio emitter and
  streaming decoder queue, evaluation judges), `lk-agents-worker` (registration, IPC,
  supervision) and `lk-agents-storage` (all database interaction, typed errors). Telemetry
  (spans, metrics, report records) is declared in ABI, recorded through `lk-agents-telemetry`
  (no executor or transport, so the sans-I/O crates can depend on it) and exported by
  `lk_agents_net::export`;
  see `docs/internals/telemetry.md`.
  `lk-agents-abi` declares the boundary once and codegen emits the Python bindings from it; pyo3
  exists only in `lk-agents-py`.
- **AgentHandler / AgentServer** (`worker.py`): a module declares an agent's doors on an
  `AgentHandler`: `rtc_session(partition_key=)` (a call, run on a `VoiceSession`) and
  `a2a_session(endpoint=, partition_key=)` (an A2A text endpoint). The entrypoint's `AgentServer` serves them
  with `server.add(...)`; the examples' `main.py` adds a front desk and a back office.
  Registration, jobs, IPC, the process pool and supervision run in `lk-agents-worker`.
- **Approval and verbatim**: `needs_approval=` holds approval in the framework (the body runs
  only on a repeat of the call after the person answered its read-back). The agent facing the
  person words questions and results unless they are `verbatim=True`.
- **VoiceSession** (`voice/agent_session.py`, decisions in `lk-agents-voice`): owns frontend expert-tool calls, an
  in-process frontend event queue, and human reply correlation without TextSession.
  Expert events are eligible when the active agent exposes the corresponding remote
  tool and the floor is available; the original Agent object need not be active.
  Expert questions are rephrased and human replies forwarded. They do not trigger
  local VoiceTask collection; that extension is outside the current contract.
  Preserve expert request identity and separate collection from reply delivery,
  allowing a future collector to explicitly answer that same pending request.
  Queue persistence is optional and separate from backend request batching. Owns
  STT, TTS, VAD, turn detection,
  endpointing, interruption and audio delivery. `update_options(...)` explicitly changes
  session configuration; selecting an agent preserves those settings. `async with
  session.override_options(...)` layers temporary settings released when the block ends.

### Key Directories
The repository is one Rust core with a binding per host language. The workspace manifests
(`pyproject.toml`, `uv.lock`) stay at the root; each language's packages live under its own
directory. Node gets `node/` beside `python/`, with `lk-agents-node` beside `lk-agents-py`.
```
rust/                   # The shared core and its bindings; see docs/internals/architecture.md
python/                 # The Python SDK
├── livekit-agents/     # livekit-agents, built with the core as livekit.agents._core
├── livekit-plugins/    # Provider plugins (openai, anthropic, google, deepgram, etc.)
├── tests/              # Test suite with fakes (fake_stt.py, fake_vad.py)
└── tests_new/          # Smoke, host, plugin and e2e suites
examples/               # Fast brain / slow brain examples: frontdesk, hotel-receptionist, airline-support
otel/                   # Generated OTel registry and consumer schema (never edited)
docs/                   # The 2.0 design, one feature per page
bench/                  # Startup, IPC, host-boundary and sentence benchmarks
```

```
python/livekit-agents/livekit/agents/   # the public Python API and host adapters
├── voice/              # Agent, VoiceSession, VoiceTask, the livekit.rtc room adapter
├── a2a.py              # The text door: A2A endpoint API, AgentClient, RemoteAgent, tokens
├── storage/            # AgentDBStorage, SQLiteStorage, thread handles, the context store
├── llm/                # Chat context, tools and toolsets, MCP, LLM and realtime APIs
├── beta/workflows.py   # Domain result types, validation helpers and transfer operations
├── stt.py, tts.py      # Speech-to-text and text-to-speech provider bases
├── inference/          # Hosted model inference (LLM, STT, TTS, realtime)
├── ipc/                # Job-process bootstrap
├── http/               # The HTTP client and the ASGI app endpoint streams are served into
├── cli.py              # dev / start subcommands invoked by `lk agent`
├── text_session.py     # Backend conversations and explicitly supplied routing state
├── evals.py            # Conversation judges (simulation context and verdicts: _generated/simulation.py)
├── telemetry.py        # OTel API provider adapter, configure, GenAI message capture, diagnostics
├── _generated/         # Generated implementations and forwarding modules (never edited)
└── _core.pyi           # Generated stubs of the Rust core extension

rust/crates/            # The Rust core, built into livekit.agents._core
├── lk-agents-abi       # Export declarations and typed boundary records
├── lk-agents-core      # AgentRun, tools/approval/output acceptance, TextSession, A2A and MCP logic
├── lk-agents-voice     # Voice session lifecycle, turns, speech scheduling, interruption
├── lk-agents-storage   # Async I/O: AgentDB and SQLite, fencing, leases, typed errors
├── lk-agents-media     # Codecs, TTS framing, images, recording files, resampling; no async runtime
├── lk-agents-telemetry # Span recording: the machines' Trace and ManualSpan, the Tracer and ScopedSpan
├── lk-agents-net       # Async I/O: transport, token signing, host stream adapter, A2A client, ProcessEnv,
│                       # telemetry export (OTLP exporter, session report upload, host span, stalls)
├── lk-agents-inference # Async I/O: provider clients, retry/fallback/reconnect, TTS emitter, judges
├── lk-agents-worker    # Async I/O: registration, jobs, IPC, process pool and supervision
├── lk-agents-cli       # Host-neutral launcher: clap, JSON logs, remote CLI and process shutdown
├── lk-agents-runtime   # The I/O layer of voice and text: executors, room I/O, drivers, HostInvoker
├── lk-agents-py        # Module registration, generated bindings; the only crate with pyo3
└── lk-agents-fuzz      # DST: sans-I/O crates and the runtime's real executors over port fakes
rust/ai_slop/           # Build tools: lk-agents-bindgen, lk-agents-host-derive, lk-agents-codegen
                        # (declarations IR, markers, generation); only their generated output is reviewed
```

### Plugin System
Plugins in `python/livekit-plugins/` provide STT, TTS, LLM, and specialized services. Each plugin is a separate package following the pattern `livekit-plugins-<provider>`. Plugins register via the `Plugin` base class in `plugin.py`. Plugins stay Python; Rust reaches them through generated proxies and HostInvoker.

### Model Interface Pattern
STT, TTS, LLM, Realtime models have provider-agnostic interfaces with:
- Base classes defining the interface, generated (`_generated/stt.py`, `tts.py`, `llm.py`, `realtime.py`)
  and exported through `stt.py`, `tts.py`, `llm/__init__.py` and `llm/realtime.py`
- Fallback adapters for resilience
- Stream adapters for different streaming patterns

### Request Flow
1. A text message reaches `/agents/<agent>/<deployment>/<endpoint>` as A2A; a call is a job dispatch.
2. The platform resolves the thread: a text's `contextId` (none creates one), or a call's job
   `thread_id`, set by its dispatch (none runs the call without a shared thread).
3. The worker takes the entrypoint's partition lease and loads the partition from the thread's database.
4. The handler runs `agent.run(...)` (text) or the `VoiceSession` runs the agent (call).
5. The framework commits history, explicitly saved context fields and task rows in one fenced
   write and releases the lease.

## Environment Variables
- `LIVEKIT_URL`: WebSocket URL of LiveKit server
- `LIVEKIT_API_KEY`: API key for authentication
- `LIVEKIT_API_SECRET`: API secret for authentication
- `LIVEKIT_AGENT_NAME`: The agent's routing name and the first segment of its endpoints
- Provider-specific keys: `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`, etc.

## Code Style
- Line length: 100 characters
- Python 3.11+ compatibility required; package support is >=3.11,<3.15 with cp311-abi3 wheels
- Google-style docstrings
- Strict mypy type checking enabled
- Use `make check` and `make fix` before committing

## Rust architecture rules

Tags: **[check]** fails in `make -C rust check` (the `rust/check` crate or codegen), **[lint]** is
Clippy (`[workspace.lints]` or a `rust/check/*/clippy.toml` policy), **[review]** is on reviewers.
`docs/internals/architecture.md` explains the design behind each rule. No rule has a baseline: fix
the violation, or allow one site with `#[allow(lint, reason = "...")]`.

### Crate graph
- Each crate depends only on the workspace crates listed for it in `rust/check/src/graph.rs`. [check]
- Dependencies flow `py -> runtime -> cores and I/O crates -> abi`, never backwards. [check]
- Only `lk-agents-py` and the DST (`lk-agents-fuzz`) depend on the runtime; core never depends on
  storage. [check]

### Sans-I/O cores (abi, core, voice, media)
- No tokio, futures, pyo3, async-trait, reqwest, hyper, tungstenite, rusqlite or LiveKit SDK in
  their resolved dependency tree; `livekit-protocol` records are fine. [check]
- Never read the clock: time comes in as an input. [lint]
- Never generate randomness or UUIDs: seeds and IDs come in as inputs. [lint]
- Never spawn tasks or threads, and never use `select!` or `join!`. [lint]
- Never read environment variables or process state. [lint]
- No locks, atomics or unbounded channels. [lint]
- Decision state uses `BTreeMap`/`BTreeSet` or `IndexMap`, never `HashMap`/`HashSet`. [lint]
- A decision subject is `mod.rs` (records), `machine` (inputs in, effects out) and `effects.rs`
  (its effect enums). [review]
- Decisions stay in the core; the runtime only executes effects. Never move a decision-bearing
  module into the runtime. [review]
- When a machine waits on two inputs that may arrive in either order, hold them in one enum that
  hands both on exactly once, never in parallel `Option` fields. [review]
- A machine waiting on a task's result learns that task's death as an input (`call: None`), so a
  lost task cannot leave it waiting. [review]
- Media's thread pool (`pool.rs`) and `media_queue` are the only exempt modules. [lint]

### Runtime executors and the DST
- Executor modules (`runtime/src/{voice,room_io,avatar,rtc}`) spawn and sleep only through the
  injected `Spawner` and `Clock`: no `tokio::spawn`, `tokio::time`, `tokio::net` or `Handle`. [check]
- Every `select!` in an executor module is `biased;`. [check]
- In the async crates, a data branch (`next()`, `recv()`) comes before a branch that runs the
  producer (`run()`), so queued data is read first; the check matches these method names. [check]
- Executor modules never call `spawn_detached`: work that outlives its caller goes through
  `owned::Background`, which the session's tasks join. [check]
- Tasks are held as `ScopedTask` handles, never ids: dropping or replacing a handle cancels its
  task, and the set still joins it. [review]
- Sockets live only in boundary adapters, which the DST replaces with port fakes. [review]
- The DST runs the runtime's real executors over faked ports, never re-implementations. [review]
- Every bug found in a real room is a DST gap: add the fake or invariant that reproduces it first. [review]
- The fakes model "committed, then the answer was lost or cancelled" and "the provider sent N
  items, then failed", not only "the operation failed". [review]
- Every run checks that each lease is released or renewed, each node input stream is closed,
  each final transcript is delivered, and nothing is applied twice. [review]
- Each cancellation bug gets a DST case that reproduces it before it is fixed. [review]

### Async I/O crates (runtime, net, inference, worker, storage, py, telemetry)
- Lock with `parking_lot`, never `std::sync::Mutex`/`RwLock`. [lint]
- Never use Tokio's blocking pool (`spawn_blocking`, `block_in_place`, `tokio::fs`, Tokio stdio):
  run blocking work on `lk_agents_net::blocking_io` or the media pool. [lint]
- Resolve names with `lk_agents_net::dns` and connect websockets with `lk_agents_net::websocket`. [lint]
- Never wrap a write or send in `timeout`: time out reads only. [review]
- Audio into a consumer goes through a `media_queue`, never a bounded channel that errors when full. [review]

### Cancellation safety
- A `select!` branch awaits only something safe to drop: `recv`, `changed`, `notified`, a sleep,
  a cancellation token, or a `&mut` to a future pinned outside the loop. [review]
- Never take an item and then await in the same branch: race readiness, take after the race. [review]
- After a branch that ends the loop wins, drain what is already queued before acting on it. [review]
- A future that has acquired something (a lease, a stream, a track, a call) runs to completion or
  holds a drop guard that releases it. [review]
- A closeable resource is held in `Owned` from the moment it exists, before the next await;
  dropping it runs its bounded close as a session cleanup task. [review]
- A timeout never cuts off a call's own cleanup: bound an operation once, not again from outside. [review]
- Retry or fall back only while nothing has been sent downstream; after the first item, a failure
  is final. [review]
- Rebuild state on reconnect from the current options, never by replaying old frames. [review]
- A waiter waits only on an `lk_agents_net::signal` cell whose `Producer` lives where the
  progress happens (moved into its task, `TaskSet::spawn_settling`, or the owning loop's table):
  dropping it resolves every waiter, `Err(Ended)` when the state can no longer come. [lint, review]
- No `Notify`, raw `watch` channel or `futures` `Shared` outside `signal.rs`: state several
  parties change is a `signal::Shared` with a terminal value every wait accepts, and work several
  waiters share is driven by one of them in its own task (a `Shared` drops a wake that lands
  while another waiter polls). [lint]
- A task feeding a machine reports its death as an input (`TaskSet::scoped_feeding`). [review]
- Every wait a host can await resolves when its session ends, drained or dropped. [review]

### Bindings and the host
- pyo3 appears only in `lk-agents-py`. [check]
- Handwritten `#[host(...)]` export declarations live only in `lk-agents-abi`; `lk-agents-py`
  declares only `#[host(interface(..))]`. Only abi, py and storage (`#[table]`) depend on
  `lk-agents-host-derive`, and ABI does not re-export `host`. [check, review]
- `#[pyclass]`, `#[pymethods]`, `#[pyfunction]` and `#[pymodule]` appear only in generated code
  and the module registration shell. [check]
- Rust calls Python only through HostInvoker (`py_calls.rs`, `py_streams.rs`) and log forwarding
  (`py_logging.rs`). [check]
- Never call the host while holding a lock; drop host objects after the guard. [review]
- Never hold a lock across `.await`; lock in-flight call state only through `GilMutex`, which
  takes a `Python` token, so the GIL is always taken first. [lint, review]
- A host call that returns a stream goes through `py_streams::Owning`, so a stream nobody took
  is closed with `aclose()`. [check]
- A type that owns an OS process kills it on drop unless it was reaped; a launch receives its
  receipt through a oneshot it owns, never a shared slot. [review]
- Every synchronous Python entry releases the GIL; only a getter may be `attached`. [check]
- Every awaited host call takes `cx: &telemetry::Cx`. Pass the enclosing span; `Cx::NONE` only
  for per-item stream calls and background loops with no owning operation. [check, review]
- A tuple crosses the boundary only as a pair of strings, bytes or numbers (a header, an env entry, a
  start/end range); anything else is a record. [check]
- Python that only forwards to `_core` is generated; handwritten Python is host-specific logic. [review]
- A plugin base class implements every proxy method. [check]
- The runtime lends the process Tokio runtime as one `ProcessEnv` (`#[host(env)] type Env;`). [review]

### Generated code
- Generated files start with `//! GENERATED FILE -- DO NOT EDIT.` and are never edited by hand. [check]
- `lk-agents-codegen --check` and `thread_api.py --check` fail on drift, missing or stale files. [check]
- A missing accessor or conversion is fixed in the schema or the generator, never by a
  handwritten copy. [review]

### Telemetry
- Spans, metrics and keys are declared once in ABI; no key is a string literal at an emitter. [review]
- Keys are unique, histogram bounds increase, and content keys sit under `lk.pii.*`. [check]
- Write `telemetry::Item`; never import telemetry names directly. [review]
- Sans-I/O crates never use `ScopedSpan`, `Tracer` or `start`/`end`/`measure`. [lint]

### Errors
- An error type distinguishes only what some caller branches on; the rest is its message. [review]
- Shared kinds: `ApiError`, `StorageError`, `HostError` (typed `HostErrorKind`), cancellation,
  and invalid input. A new variant needs a caller that matches on it. [review]
- No `Result<_, String>`, and never decide anything from an error's message. [review]
- Only a storage connection that sent a write calls `Unacknowledged::sent`; every other
  `Uncertain` is passed on, never made. [check]
- Every database write goes through `Resend` with a key that makes resending safe: an uncertain
  write is retried with the same key. [review]
- A write without a lease is guarded on its run's `run_ended`, so applying it twice writes
  nothing more. [review]
- A re-sent write that finds itself already applied reports success. [review]
- `Uncertain` reaches the caller only when retries ran out and the outcome is still unknown. [review]
- Local pressure is never a non-retryable error. [review]

### Idiomatic Rust
- Every crate uses `[workspace.lints]`; the workspace builds with zero warnings. [lint]
- No `.unwrap()` outside tests and fuzz: use `?`, a typed error, or `expect("why it holds")`. [lint]
- Every `#[allow]` carries `reason = "..."`. [lint]
- Every `unsafe` block has a `// SAFETY:` comment; unsafe stays in narrow FFI and codec adapters. [lint]
- No `dbg!`, `todo!`, `unimplemented!` or `println!` in library code. [lint]
- Use `name/mod.rs` for a module with children; never `name.rs` beside a `name/` directory. [lint]
- Use fields, not getters and setters, unless a method protects an invariant. [review]
- Name events and effects for their domain (`VoiceEffect`), never bare `Effect`. [review]
- Closed sets are ABI enums with generated strings, never string compares. [review]

### Layout and tests
- Handwritten `.rs` files stay within 1,250 lines; split by responsibility. [check]
- Out-of-line tests live in sibling `*_tests.rs` files named for their subject; keep small tests
  inline. [review]
- Python binding contracts live beside their implementation in `lk-agents-py/src/*_tests.py`
  and run through the `py_bindings` Cargo target:
  `uv run --no-sync cargo test --manifest-path rust/Cargo.toml -p lk-agents-py --features ffmpeg`. [review]
- Rust tests load host fixtures from `py_test_fixtures.py` through the shared loader; never
  assemble Python source strings or use `eval`. [review]
