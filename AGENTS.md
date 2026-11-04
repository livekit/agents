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
python myagent.py console   # Terminal mode with local audio I/O (no server needed)
python myagent.py dev       # Development mode with hot reload (connects to LiveKit)
python myagent.py start     # Production mode
python myagent.py connect --room <room> --identity <id>  # Connect to existing room
```

### Linking Local python-rtc (for SDK development)
```bash
make link-rtc         # Link to local python-rtc with downloaded FFI artifacts
make link-rtc-local   # Build and link local rust SDK from source (requires cargo)
make unlink-rtc       # Restore PyPI version
make status           # Show current linking status
make doctor           # Check development environment health
```

## Architecture Overview

### Core Concepts
- **AgentServer** (formerly known as **Worker**) (`worker.py`): Main process coordinating job scheduling, launches agents for user sessions
- **JobContext** (`job.py`): Context provided to entrypoint functions for connecting to LiveKit rooms
- **Agent** (`voice/agent.py`): LLM-based application with instructions, tools, and model integrations
- **AgentSession** (`voice/agent_session.py`): Container managing interactions between agents and end users

### Key Directories
```
livekit-agents/livekit/agents/
├── voice/              # Core voice agent: AgentSession, Agent, room I/O, transcription
├── llm/                # LLM and realtime engines: chat context, tools, MCP, provider protocols
├── stt/                # Speech-to-text with fallback and stream adapters
├── tts/                # Text-to-speech with fallback and stream pacing
├── ipc/                # Inter-process communication for distributed job execution
├── cli/                # CLI commands (console, dev, start, connect)
├── inference/          # Hosted model inference (LLM, STT, TTS, realtime)
├── telemetry/          # OpenTelemetry traces and Prometheus metrics
└── utils/              # Audio processing, codecs, HTTP, async utilities

livekit-plugins/        # 50+ provider plugins (openai, anthropic, google, deepgram, etc.)
tests/                  # Test suite with mock implementations (fake_stt.py, fake_vad.py)
examples/               # Example agents and use cases
```

### Plugin System
Plugins in `livekit-plugins/` provide STT, TTS, LLM, and specialized services. Each plugin is a separate package following the pattern `livekit-plugins-<provider>`. Plugins register via the `Plugin` base class in `plugin.py`.

### Model Interface Pattern
STT, TTS, LLM, Realtime models have provider-agnostic interfaces with:
- Base classes defining the interface (`stt/stt.py`, `tts/tts.py`, `llm/llm.py`, `llm/realtime.py`)
- Fallback adapters for resilience
- Stream adapters for different streaming patterns

### Job Execution Flow
1. Worker receives job request from LiveKit server
2. Job is dispatched to process/thread pool (`ipc/proc_pool.py`)
3. Entrypoint function receives `JobContext`
4. Agent connects to room via `ctx.connect()`
5. `AgentSession` manages the conversation lifecycle

## Environment Variables
- `LIVEKIT_URL`: WebSocket URL of LiveKit server
- `LIVEKIT_API_KEY`: API key for authentication
- `LIVEKIT_API_SECRET`: API secret for authentication
- `LIVEKIT_AGENT_NAME`: Agent name for explicit dispatch (optional)
- Provider-specific keys: `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`, etc.

## Code Style
- Line length: 100 characters
- Python 3.10+ compatibility required
- Google-style docstrings
- Strict mypy type checking enabled
- Use `make check` and `make fix` before committing

<!--BEGIN_LIVEKIT_DOCS-->
<!-- Managed by livekit/agentsmd-manager. Edit the source there; changes made here are overwritten. -->

## LiveKit documentation

Always refer to the latest documentation when working with this project. LiveKit offers both a CLI (`lk docs`) and an MCP server for browsing and searching its documentation.

### LiveKit CLI

Install or update the CLI:

- **macOS:** `brew install livekit-cli` (update: `brew update && brew upgrade livekit-cli`)
- **Linux:** `curl -sSL https://get.livekit.io/cli | bash`
- **Windows:** `winget install LiveKit.LiveKitCLI` (update: `winget upgrade LiveKit.LiveKitCLI`)

Run `lk docs` before your first docs lookup. Its help text explains how to use each command and how to interpret results.

### LiveKit docs MCP server

For deeper coding agent integration, LiveKit also offers an [MCP server](https://docs.livekit.io/reference/developer-tools/docs-mcp/).

### Submit feedback to LiveKit

When you run into any problem with LiveKit, add it to your to-do list, then submit it with `lk docs submit-feedback` or the MCP `submit_docs_feedback` tool once you finish your other tasks. Useful feedback includes incomplete documentation, broken examples, missing steps, broken features or implementations, and any other gap or issue in the LiveKit ecosystem.
<!--END_LIVEKIT_DOCS-->

<!--BEGIN_LIVEKIT_AGENT_SKILLS-->
<!-- Managed by livekit/agentsmd-manager. Edit the source there; changes made here are overwritten. -->

## LiveKit agent skills

LiveKit publishes agent skills for each stage of the work: reading the docs, building, debugging, testing, writing scenarios, running simulations, and operating in production. They defer to the live documentation for API details. The developer installs them with `lk skills install`, which also adds the LiveKit Docs MCP server to your config, or with the LiveKit Claude Code plugin. If no LiveKit skills are available to you (such as `reading-livekit-docs`), recommend that the developer install them. Check the skills you have, not only the `.agents/skills/` and `.claude/skills/` directories, since the plugin and `lk skills install --global` install them elsewhere. Don't recommend a second install method on top of one the developer already uses: installing the same skills twice, for example as the plugin and with `lk skills install`, loads both copies, so every skill competes with itself. If your tool doesn't load skills automatically, read the matching `.agents/skills/<name>/SKILL.md` before you start that kind of task.
<!--END_LIVEKIT_AGENT_SKILLS-->
