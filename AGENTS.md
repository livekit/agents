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

### Testing
```bash
uv run pytest --unit                    # Run all unit tests
uv run pytest tests/test_tools.py       # Run a single test file
make unit-tests                         # Run unit tests that don't require cloud accounts
```

#### Test categories

Every test module declares exactly one category via a module-level marker, and
each category has a matching `--<category>` selection flag. Selection happens
*before* import, so a category run never imports (or fails on) modules outside
it.

| Marker | Flag | Meaning |
|--------|------|---------|
| `pytest.mark.unit` | `--unit` | fast, hermetic, no external providers/credentials/network |
| `pytest.mark.audio_eot` | `--audio_eot` | hermetic audio end-of-turn / turn-detection suite |
| `pytest.mark.plugin("name")` | `--plugin [name]` | provider integration test (needs that provider's deps/keys) |
| `pytest.mark.stt` | `--stt` | cross-provider speech-to-text suite (`tests/test_stt.py`) |
| `pytest.mark.tts` | `--tts` | cross-provider text-to-speech suite (`tests/test_tts.py`) |
| `pytest.mark.realtime("name")` | `--realtime [name]` | realtime-model test |
| `pytest.mark.evals` | `--evals` | behavioral evals against the LiveKit inference gateway |
| `pytest.mark.docs` | `--docs` | tests for the docs-build tooling under `.github/` |

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
python myagent.py dev       # Development mode with hot reload (connects to LiveKit)
python myagent.py start     # Production mode
```
These subcommands are the interface `lk agent dev|start|console` (the Go livekit-cli) invokes.

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
per page; `planned-*.md` pages describe what is not built yet, and `docs/TODO.md` lists where the
code still differs from a built page.

### Core Concepts
- **Fast brain / slow brain**: the pattern every example follows. A frontend agent talks to the
  person (persona, pacing, voice) and its one tool is an `A2ADelegate` to a backend agent that
  holds the tools, the state and the systems of record.
- **Agent** (`voice/agent.py`): configuration (instructions, models, tools, `deps_type`,
  `state_type`, `output_type`), built once at import and shared. `await agent.run(message,
  message_history=, state=, deps=)` is one turn and returns a `RunResult`.
- **RunContext[Deps, State]**: what tools and `@agent.instructions` receive: `ctx.deps` (per run,
  never saved), `ctx.state` (saved on the thread), `ctx.thread`, `ctx.live_session`, `ctx.update()`.
- **Transitions** (`InputRequired`, `Completed`, `Failed`, `Rejected`, `Working`): what a tool
  returns to say what happens next; `InputRequired` waits for the person.
- **AgentTask[Output, State]**: a multi-turn flow awaited from a tool, on a call and in text.
- **Thread**: one durable conversation (`AT_...` id, the A2A `contextId`), with one lane per
  agent holding its history, state, agent stack and tasks; stored in agent-db.
- **AgentHandler / AgentServer** (`worker.py`): a module declares an agent's doors on an
  `AgentHandler(lane=)`: `rtc_session()` (a call, run on a `LiveSession`) and
  `a2a_session(endpoint=)` (an A2A text endpoint). The entrypoint's `AgentServer` serves them
  with `server.add(...)`; the examples' `main.py` adds a front desk and a back office.
- **Approval and verbatim**: `requires_approval=` holds approval in the framework (the body runs
  only on a repeat of the call after the person answered its read-back). The agent facing the
  person words questions and results unless they are `verbatim=True`.
- **LiveSession** (`voice/agent_session.py`): the runtime a call's room attaches to: models, turn
  detection, audio. `AgentSession` is a deprecated alias kept one release.

### Key Directories
```
livekit-agents/livekit/agents/
├── voice/              # Agent, LiveSession, AgentTask, room I/O, transcription
├── a2a/                # The text door: A2A endpoint, codec, LiveKit extension, AgentClient, tokens
├── delegation/         # Delegate, A2ADelegate (the delegate toolset), record_delegation
├── storage/            # AgentDBStorage, thread resolution, the call's lane (live.py)
├── graph.py            # BaseNode, End, Graph
├── llm/                # Chat context, tools and toolsets, MCP, LLM and realtime engines
├── beta/workflows/     # Prebuilt AgentTasks (email, phone, address, credit card, DOB, TaskGroup)
├── stt/, tts/          # Speech-to-text and text-to-speech, fallback and stream adapters
├── inference/          # Hosted model inference (LLM, STT, TTS, realtime)
├── ipc/                # Process pool, job processes, the ASGI app endpoint streams are served into
├── cli/                # dev / start subcommands invoked by `lk agent`
├── simulation.py       # Scenarios for `lk agent simulate`
├── telemetry/          # OpenTelemetry traces and Prometheus metrics
└── _core.pyi           # Stubs of the Rust core extension

rust/crates/            # The Rust core (worker control plane, agent-db client, voice driver,
                        # inference clients, media), built into livekit.agents._core; see docs/rust-core.md
livekit-plugins/        # Provider plugins (openai, anthropic, google, deepgram, etc.)
examples/               # Fast brain / slow brain examples: frontdesk, hotel-receptionist, airline-support
docs/                   # The 2.0 design, one feature per page; docs/examples/use-cases/ one agent per idea
console/                # Next.js web console: chat, call, and a live view of the thread
skills/                 # Agent skills shipped with the repo (migrate-to-v2)
tests/                  # Test suite with fakes (fake_stt.py, fake_vad.py)
```

### Plugin System
Plugins in `livekit-plugins/` provide STT, TTS, LLM, and specialized services. Each plugin is a separate package following the pattern `livekit-plugins-<provider>`. Plugins register via the `Plugin` base class in `plugin.py`. Plugins stay Python; the Rust core reaches them through host proxies.

### Model Interface Pattern
STT, TTS, LLM, Realtime models have provider-agnostic interfaces with:
- Base classes defining the interface (`stt/stt.py`, `tts/tts.py`, `llm/llm.py`, `llm/realtime.py`)
- Fallback adapters for resilience
- Stream adapters for different streaming patterns

### Request Flow
1. A text message reaches `/agents/<agent>/<deployment>/<endpoint>` as A2A; a call is a job dispatch.
2. The platform resolves the thread (the `contextId`, or `thread_id` in the dispatch metadata) or creates one.
3. The worker loads the agent's lane from agent-db and takes its lease.
4. The handler runs `agent.run(...)` (text) or the `LiveSession` runs the agent (call).
5. The framework commits the lane (history, state, agent stack) and releases the lease.

## Environment Variables
- `LIVEKIT_URL`: WebSocket URL of LiveKit server
- `LIVEKIT_API_KEY`: API key for authentication
- `LIVEKIT_API_SECRET`: API secret for authentication
- `LIVEKIT_AGENT_NAME`: The agent's name: its lane in a thread and the first segment of its endpoints
- Provider-specific keys: `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`, etc.

## Code Style
- Line length: 100 characters
- Python 3.10+ compatibility required
- Google-style docstrings
- Strict mypy type checking enabled
- Use `make check` and `make fix` before committing
