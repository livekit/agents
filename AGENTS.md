# AGENTS.md

## Build and Development Commands

This project uses **uv** as the package manager. All commands run from the repository root.

### Installation
```bash
make install          # Install all dependencies with dev extras (uv sync --all-extras --dev)
```

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
make -C rust clippy        # Shared purity policy for ABI, core, voice and media
make -C rust architecture  # Full architecture gate, fixtures and codegen checks
make -C rust check         # Rust formatting and the architecture gate
```

### Testing
```bash
uv run pytest --unit                    # Run all unit tests
uv run pytest tests/test_tools.py --unit # Run a single unit test file
make unit-tests                         # Run unit tests that don't require cloud accounts
```

#### Test categories

Every test module declares exactly one category via a module-level marker, and
each category has a matching `--<category>` selection flag. Selection happens
*before* test-module import, so unselected test modules are not imported.
Shared `conftest.py` fixtures and their dependencies still load. Categories are
registered in `pyproject.toml`; `tests/conftest.py` reads their literal markers.
Put test paths before category flags, whose optional argument is a provider name.

| Marker | Flag | Meaning |
|--------|------|---------|
| `pytest.mark.unit` | `--unit` | fast, hermetic, no external providers/credentials/network |
| `pytest.mark.audio_eot` | `--audio_eot` | hermetic audio end-of-turn / turn-detection suite |
| `pytest.mark.plugin("name")` | `--plugin [name]` | provider integration test (needs that provider's deps/keys) |
| `pytest.mark.stt` | `--stt` | cross-provider speech-to-text suite; requires provider dependencies and credentials |
| `pytest.mark.tts` | `--tts` | cross-provider text-to-speech suite; requires provider dependencies and credentials |
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
`lk agent` (the Go livekit-cli) runs `python main.py start [--dev]`, `console` or `download-files`;
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
- **Agent** (`voice/agent.py`): shared configuration, `Agent[Context]`, and validated
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
  shared agent configuration and explicitly supplied saved state.
  Each child requires `output_type`; accepted final output fills the parent tool call.
  Returns `result: RunResult` and `state: TextSessionState`; application code explicitly
  loads and saves backend routing state. See `docs/text-session.md`.
- **VoiceTask[Output]**: fast-brain/frontend live interactions in an active voice session.
  `.run()` resolves the session from invocation-local runtime context; callers do not
  pass it explicitly. Calling outside a live interaction fails before execution.
  Temporarily takes the floor, returns a typed result, and restores the parent without
  calling the parent's on_exit/on_enter hooks. Its
  parent await remains in memory. Not for stateless workloads, backend orchestration,
  background execution, or recovery across requests/deployments. See `docs/voice-tasks.md`.
- **Context persistence**: `persisted(...)` marks fields; `load_context` and `save_context`
  are explicit. Runtime services remain application-supplied objects.
- **Thread**: one conversation (`AT_...` id, the A2A `contextId`), with configured handler lanes,
  histories and transport tasks stored in its database (agent-db, or local SQLite).
- **Rust core** (`rust/crates/`, see `docs/internals/rust-core.md`): the sans-I/O crates
  (`lk-agents-core` with its `a2a` and MCP logic, `-voice`, `-media`) are synchronous (events and
  time in, effects out); media is CPU work with no async runtime. `lk-agents-runtime` is only the
  I/O layer of voice and text: the voice executor, room I/O, avatar transport, recording
  ingress, AgentRun/TextSession drivers, the thin A2A door (executes core's storage effects by
  calling storage), thin MCP I/O, HostInvoker and the process runtime; it makes no decisions.
  Everything that can be faked at its boundary is ordinary async Rust, tested on its own and
  faked in the DST: `lk-agents-net` (the shared HTTP/SSE/WebSocket transport, the host stream
  adapter, the delegation A2A client and the Cloud session report upload),
  `lk-agents-inference` (providers, retry/fallback/reconnect, the TTS audio emitter and
  streaming decoder queue, evaluation judges), `lk-agents-worker` (registration, IPC,
  supervision) and `lk-agents-storage` (all database interaction, typed errors). Telemetry is
  recorded synchronously and reaches the host's OpenTelemetry SDK through HostInvoker.
  `lk-agents-abi` declares the boundary once and codegen emits the Python bindings from it; pyo3
  exists only in `lk-agents-py`.
- **AgentHandler / AgentServer** (`worker.py`): a module declares an agent's doors on an
  `AgentHandler(lane=)`: `rtc_session()` (a call, run on a `VoiceSession`) and
  `a2a_session(endpoint=)` (an A2A text endpoint). The entrypoint's `AgentServer` serves them
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
```
livekit-agents-next/livekit/agents/   # the public Python API and host adapters
├── voice/              # Agent, VoiceSession, VoiceTask, the livekit.rtc room adapter
├── a2a/                # The text door: A2A endpoint API, AgentClient, RemoteAgent, tokens
├── storage/            # AgentDBStorage, SQLiteStorage, thread handles, the context store
├── llm/                # Chat context, tools and toolsets, MCP, LLM and realtime APIs
├── beta/workflows.py   # Domain result types, validation helpers and transfer operations
├── stt.py, tts.py      # Speech-to-text and text-to-speech provider bases
├── inference/          # Hosted model inference (LLM, STT, TTS, realtime)
├── ipc/                # Job-process bootstrap and the ASGI app endpoint streams are served into
├── cli.py              # dev / start subcommands invoked by `lk agent`
├── text_session.py     # Backend conversations and explicitly supplied routing state
├── evals.py            # Conversation judges
├── simulation.py       # Simulation context and application verdicts
├── telemetry/          # OpenTelemetry traces and Prometheus metrics
├── _generated/         # Generated implementations and forwarding modules (never edited)
└── _core.pyi           # Generated stubs of the Rust core extension

rust/crates/            # The Rust core, built into livekit.agents._core; see docs/internals/rust-core.md
├── lk-agents-abi       # Export declarations and typed boundary records
├── lk-agents-core      # AgentRun, tools/approval/output acceptance, TextSession, A2A and MCP logic
├── lk-agents-voice     # Voice session lifecycle, turns, speech scheduling, interruption
├── lk-agents-storage   # Async I/O: AgentDB and SQLite, fencing, leases, typed errors
├── lk-agents-media     # Codecs, TTS framing, images, recording files, resampling; no async runtime
├── lk-agents-net       # Async I/O: transport, token signing, host stream adapter, A2A client, ProcessEnv
├── lk-agents-inference # Async I/O: provider clients, retry/fallback/reconnect, TTS emitter, judges
├── lk-agents-worker    # Async I/O: registration, jobs, IPC, process pool and supervision
├── lk-agents-runtime   # The I/O layer of voice and text: executors, room I/O, drivers, HostInvoker
├── lk-agents-py        # Module registration, generated bindings; the only crate with pyo3
├── lk-agents-fuzz      # DST: sans-I/O crates and the runtime's real executors over port fakes
└── lk-agents-bench-steps  # Voice data-plane step measurements (compiled in the launch gate)
rust/ai_slop/           # Build tools: lk-agents-bindgen, lk-agents-host-derive, lk-agents-codegen
                        # (declarations IR, markers, generation); only their generated output is reviewed
livekit-plugins/        # Provider plugins (openai, anthropic, google, deepgram, etc.)
examples/               # Fast brain / slow brain examples: frontdesk, hotel-receptionist, airline-support
docs/                   # The 2.0 design, one feature per page
console/                # Next.js web console: chat, call, and a live view of the thread
skills/                 # Agent skills shipped with the repo (migrate-to-v2)
tests/                  # Test suite with fakes (fake_stt.py, fake_vad.py)
```

### Plugin System
Plugins in `livekit-plugins/` provide STT, TTS, LLM, and specialized services. Each plugin is a separate package following the pattern `livekit-plugins-<provider>`. Plugins register via the `Plugin` base class in `plugin.py`. Plugins stay Python; Rust reaches them through generated proxies and HostInvoker.

### Model Interface Pattern
STT, TTS, LLM, Realtime models have provider-agnostic interfaces with:
- Base classes defining the interface (`stt/stt.py`, `tts/tts.py`, `llm/llm.py`, `llm/realtime.py`)
- Fallback adapters for resilience
- Stream adapters for different streaming patterns

### Request Flow
1. A text message reaches `/agents/<agent>/<deployment>/<endpoint>` as A2A; a call is a job dispatch.
2. The platform resolves the thread (the `contextId`, or `thread_id` in the dispatch metadata) or creates one.
3. The worker takes the agent's lane lease and loads the lane from the thread's database.
4. The handler runs `agent.run(...)` (text) or the `VoiceSession` runs the agent (call).
5. The framework commits history, explicitly saved context fields and task rows in one fenced
   write and releases the lease.

## Environment Variables
- `LIVEKIT_URL`: WebSocket URL of LiveKit server
- `LIVEKIT_API_KEY`: API key for authentication
- `LIVEKIT_API_SECRET`: API secret for authentication
- `LIVEKIT_AGENT_NAME`: The agent's name: its lane in a thread and the first segment of its endpoints
- Provider-specific keys: `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`, etc.

## Code Style
- Line length: 100 characters
- Python 3.11+ compatibility required; package support is >=3.11,<3.15 with cp311-abi3 wheels
- Google-style docstrings
- Strict mypy type checking enabled
- Use `make check` and `make fix` before committing

## Rust architecture rules
- Dependency direction is `lk-agents-py -> lk-agents-runtime -> sans-I/O and async I/O crates ->
  lk-agents-abi`; `voice -> core`, `voice -> media`, `inference -> media`, `inference -> core`,
  `net -> core`, `py -> abi`, async I/O crates `-> net`, `fuzz` and
  `bench-steps -> sans-I/O crates + abi`, `fuzz -> runtime` (the DST runs the runtime's real
  executors over faked ports, the fake room among them) and `fuzz -> storage` (the A2A DST runs
  real storage) are allowed composition edges.
  `lk-agents-py` depends only on the runtime and ABI (the runtime re-exports the
  implementations generated adapters select). ABI never depends on subjects or runtime; cores
  and async I/O crates never depend on runtime, `lk-agents-py` or host adapters. Core never
  depends on `lk-agents-storage`: its A2A decisions use ABI-declared storage records and errors,
  and the runtime's A2A door executes them through storage.
- The sans-I/O crates (`lk-agents-core`, `-voice`, `-media`) and `lk-agents-abi`
  have no executor, async transport or SDK dependency in their resolved dependency closure:
  no tokio, futures, pyo3, async-trait, reqwest, tungstenite, rusqlite, `livekit` or
  `livekit-ffi`. `livekit-protocol` records remain allowed. The check resolves abi, core, voice
  and media with their own features. These dependency bans do not apply to runtime, the async
  I/O crates (`lk-agents-net`, `-inference`, `-worker`, `-storage`), fuzz support, language
  bindings or build tools.
- pyo3 exists only in `lk-agents-py`. Every Rust-to-host call goes through HostInvoker with
  the Invocation captured at its operation's entry and its declared scopes; async I/O crates
  hold only ABI-declared proxy traits. Log forwarding is the one host entry outside
  HostInvoker: `log` lines are queued without the GIL and `lk-agents-py`'s drain thread forwards
  them to Python's `logging`, so logging anywhere, under any lock, never waits for the GIL.
- The I/O crates and py lock only through `lk_agents_net::lock`: a thread holding the GIL
  waits for a contended lock with the GIL released, so a host call under a lock (a log line, a
  proxy, an emit) cannot deadlock. The I/O Clippy policy (`rust/architecture/io/clippy.toml`)
  bans raw `Mutex`/`RwLock` types there.
- Rust has no telemetry exporter: observations are recorded synchronously and reach the host's
  OpenTelemetry SDK through HostInvoker. The LiveKit Cloud session report and recording upload
  is not user telemetry; it is async Rust in `lk-agents-net`.
- The runtime owns the process Tokio runtime, fork guard and media pool and lends them as one
  `lk_agents_net::env::ProcessEnv` value: a declared class declares `#[host(env)] type Env;` and
  the generated binding passes the runtime's value to it; no `Env` type parameters. Its only
  functions from another layer are the voice core's text algorithms, lent as plain functions.
- Runtime executor modules (`voice/`, `room_io/`, `avatar/`, `rtc/`) use no Tokio runtime
  (`tokio::spawn`, `tokio::time`, `tokio::net`, `Handle`, `Runtime`): they spawn and sleep only
  through the injected `Spawner` and `Clock`, so the DST runs them on its deterministic
  scheduler. Tokio sync primitives are allowed; every `select!` is `biased;`. Sockets live in
  boundary adapters (the SDK room, the console's TCP port) that the DST replaces with port fakes.
- Move executors into the runtime by splitting: decisions stay in voice/core as synchronous
  state machines emitting effects; the runtime keeps pumps, node calls, timers and output
  binding. Never move a decision-bearing module into the runtime wholesale.
- The shared `rust/architecture/clippy.toml`, explicitly selected for core, voice, ABI and media,
  uses `disallowed-methods`, `disallowed-types` and `disallowed-macros` to reject spawns,
  `select!`/`join!`, ambient clocks/entropy/environment/process reads,
  unbounded channels, locks, host-shared atomics and randomly ordered decision state.
  Media's optional thread pool, added only for a measured need, is its one allowed spawn
  site. Storage, inference, worker and net are unrestricted async crates.
  Supply time and IDs as inputs; keep live resources and execution in runtime or the async I/O
  crates.
- Every git-tracked handwritten `.rs` file stays within 1,250 lines; files with the generated
  marker are skipped. Split by responsibility when moving code.
- Handwritten export declarations (`#[host]`, `#[derive(Host)]`, `#[class]`, `#[binding]` and
  export metadata) live only in `lk-agents-abi`. Handwritten `#[pyclass]`, `#[pymethods]`,
  `#[pyfunction]` and `#[pymodule]` belong only in generated output or the module registration shell.
  Generated files are never edited manually; `lk-agents-codegen --check` rejects placement violations
  and generated drift.
- Existing violations have checked-in baselines; clippy exemptions also carry `#[allow(...)]`
  with a reason. Baselines shrink by violation kind and location identity, never by count:
  removing one violation does not permit another. Never add or broaden an exemption.
- Run `rust/architecture/check.sh` for dependency, clippy, baseline and file-length enforcement;
  CI and the launch gate use the same step. Every check has deliberately failing fixtures.
