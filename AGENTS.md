# AGENTS.md

This file is the working map for AI agents and contributors in this repository.
The main target is `livekit/agents`, the Python framework for realtime voice and
multimodal AI agents. The sibling `livekit` server repository can be used as a
reference for server-side agent dispatch, but day-to-day changes should happen
here unless the protocol or server behavior itself must change.

## What This Project Does

LiveKit Agents lets developers run programmable server-side participants in
LiveKit rooms. A typical agent listens to realtime audio/video/data from a room,
uses STT, VAD, LLM, Realtime APIs, TTS, tools, and handoffs, then publishes audio,
text, or data back into the same room. It is designed for production voice agents,
telephony agents, avatars, workflows, and testable LLM applications.

The repository is a Python monorepo:

- `livekit-agents/` contains the core `livekit.agents` package.
- `livekit-plugins/` contains provider packages for OpenAI, Deepgram, Cartesia,
  Google, Anthropic, Silero, turn detection, avatars, telephony helpers, and many
  other integrations.
- `examples/` contains runnable agents and integration examples.
- `tests/` contains core unit tests, fake model implementations, plugin tests,
  realtime tests, and docker-based provider tests.
- `scripts/` contains repo tooling, especially type-check orchestration.

## Tooling And Commands

Use `uv` from the repository root.

```bash
make install                 # uv sync --all-extras --dev
make fix                     # ruff format + ruff check --fix
make check                   # ruff format --check + ruff check + mypy
make type-check              # python scripts/check_types.py
make unit-tests              # curated unit-test list from the root makefile
uv run pytest tests/test_tools.py
uv run pytest tests/test_tools.py -k test_name
uv build --package livekit-agents
uv build --package livekit-plugins-openai
```

`tests/Makefile` has heavier docker/provider workflows:

```bash
cd tests && make unit-tests
cd tests && make test PLUGIN=openai
cd tests && make realtime-tests
```

Run example agents from the repo root:

```bash
uv run examples/voice_agents/basic_agent.py console
uv run examples/voice_agents/basic_agent.py dev
uv run examples/voice_agents/basic_agent.py start
uv run examples/voice_agents/basic_agent.py connect --room my-room
```

`console` runs locally with terminal/audio I/O. `dev`, `start`, and `connect`
need LiveKit connection settings:

```bash
LIVEKIT_URL=wss://...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

Provider plugins use provider-specific environment variables, such as
`OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`, `CARTESIA_API_KEY`,
and similar names used by each plugin.

## Package Map

Core package: `livekit-agents/livekit/agents/`

- `worker.py`: `AgentServer` and `WorkerOptions`. Owns worker lifecycle,
  LiveKit server registration, job availability, assignments, termination,
  load reporting, health endpoints, process pools, and CLI runtime integration.
- `job.py`: `JobContext`, `JobRequest`, `JobProcess`, room connection helpers,
  shutdown hooks, SIP helpers, participant entrypoints, recording setup, and
  session report upload.
- `voice/agent_session.py`: public `AgentSession`; the top-level runtime for a
  user interaction. It coordinates room I/O, model instances, turn handling,
  interruption behavior, tools, metrics, recording, and session lifecycle.
- `voice/agent.py`: public `Agent`; instructions, tools, per-agent STT/VAD/LLM/TTS
  overrides, handoff hooks, and overridable pipeline nodes.
- `voice/agent_activity.py`: internal state machine for a running agent. It handles
  audio recognition, end-of-turn, preemptive generation, queued speech, tool
  execution, handoffs, interruption recovery, and realtime-model behavior.
- `voice/generation.py`: LLM, tool, text, TTS, and audio forwarding helpers used
  by `AgentActivity`.
- `voice/audio_recognition.py`, `voice/endpointing.py`, `voice/turn.py`: VAD/STT
  recognition, endpointing, interruption, and turn-detection behavior.
- `voice/room_io/`: `RoomIO`, input, output, pre-connect audio, transcription,
  and RTC room media plumbing.
- `voice/io.py`: provider-neutral audio/video/text I/O abstractions.
- `llm/`: provider-neutral LLM, realtime model, chat context, function tool,
  MCP, toolset, fallback, and provider-format abstractions.
- `stt/`, `tts/`, `vad.py`: provider-neutral model interfaces, streaming adapters,
  fallback adapters, speech events, synthesized audio events, and metrics.
- `inference/`: LiveKit Inference clients for cloud-routed STT/LLM/TTS.
- `ipc/`: process/thread executors, process supervision, socket IPC, lazy child
  process entrypoints, inference executors, and mock rooms.
- `cli/`: Typer CLI for `console`, `dev`, `start`, `connect`, and `download-files`.
- `telemetry/`, `metrics/`, `observability.py`: OpenTelemetry, Prometheus, model
  usage, traces, and LiveKit Cloud observability integration.
- `evals/`: evaluation and judge helpers.
- `beta/`: newer toolsets and workflow helpers that may still be changing.
- `utils/`: async primitives, audio helpers, codecs, HTTP context, CPU monitoring,
  image helpers, logging, deprecation helpers, and general utilities.

Plugin packages live under `livekit-plugins/livekit-plugins-*/livekit/plugins/*`.
Most plugin packages follow this shape:

- `pyproject.toml` for package metadata and provider dependencies.
- `livekit/plugins/<provider>/__init__.py` exporting public classes and calling
  `Plugin.register_plugin(...)`.
- Provider implementations for one or more of `LLM`, `RealtimeModel`, `STT`,
  `TTS`, `VAD`, avatar, or workflow interfaces.
- `models.py` for typed model names and provider options.
- `version.py`, `log.py`, and `py.typed`.
- Plugin-specific tests in `tests/` or package-local test files for native/C
  extension packages.

## Runtime Architecture

Production dispatch flow:

1. User code creates an `AgentServer` or `WorkerOptions` and registers exactly one
   RTC entrypoint with `@server.rtc_session(...)`.
2. `cli.run_app(server)` exposes `console`, `dev`, `start`, `connect`, and
   `download-files`.
3. `AgentServer.run()` starts an HTTP health server, optional Prometheus server,
   an optional inference executor, and an `ipc.ProcPool`.
4. The worker connects to the LiveKit server `/agent` WebSocket with a JWT that
   has the `agent` grant.
5. The worker sends `RegisterWorkerRequest` containing job type, agent name,
   permissions, and SDK version.
6. LiveKit server sends availability requests. `_answer_availability()` checks
   load and calls the user `on_request` handler, which accepts or rejects the job.
7. On accept, LiveKit server sends assignment with room URL and token.
8. `ProcPool.launch_job()` runs the job in a warmed process by default, or a
   thread executor on Windows / when configured.
9. The entrypoint receives `JobContext`. It usually calls `ctx.connect()`,
   starts an `AgentSession`, and optionally waits for participants or registers
   participant entrypoints.
10. Job status updates are reported back to LiveKit server until shutdown,
    success, failure, or termination.

Local modes:

- `console` creates a fake/local job and uses terminal or sounddevice I/O. It is
  best for fast agent logic iteration without LiveKit server dispatch.
- `dev` connects to LiveKit and enables development defaults plus file watching.
- `connect` starts unregistered, creates or finds a room, then simulates a job for
  that room.
- `start` is production mode.

Voice turn flow:

1. `RoomIO` or custom I/O produces audio/video/text input.
2. `AgentSession` and `AgentActivity` choose the active `Agent` and model stack.
3. Audio recognition uses VAD/STT or realtime model events to detect speech and
   end-of-turn.
4. `Agent.on_user_turn_completed()` may inspect or mutate the pending turn.
5. The LLM/realtime model generates text and/or tool calls.
6. Tool calls execute through `FunctionTool`, `Toolset`, MCP toolsets, or provider
   tools; tool outputs may trigger more LLM steps or handoffs.
7. TTS converts assistant text to audio unless a realtime model already supplies
   audio.
8. Output is forwarded through room I/O, transcription synchronization, and event
   callbacks.
9. Metrics, traces, usage, recording, and session reports are emitted around the
   flow.

## Server Reference

The Go `livekit-server` repository is useful when a question touches dispatch
protocol or server behavior. Relevant reference files there:

- `pkg/service/agentservice.go`: `/agent` WebSocket endpoint, worker handshake,
  worker registry, job request handling.
- `pkg/agent/worker.go`: server-side worker protocol, register/availability/job
  update/ping/migration handling.
- `pkg/agent/client.go`: server-side client that launches or terminates jobs on
  registered workers.
- `pkg/service/agent_dispatch_service.go`: public dispatch API.
- `pkg/rtc/room.go`: room lifecycle points that launch room, publisher, and
  participant agent jobs.

Protocol objects come from `livekit-protocol`. Prefer changing only this repo
unless the protocol contract or server dispatch behavior must change.

## Contribution Guide

When touching public API:

- Update exports in `livekit-agents/livekit/agents/__init__.py` if the symbol is
  meant to be public.
- Add or update docstrings. Contributing docs require documented public methods,
  classes, and enums because API docs are generated.
- Preserve backward compatibility where practical. Use
  `utils/deprecation.py` patterns for renamed parameters and note target removal
  versions.
- Add focused tests for the behavior. Do not rely only on examples.
- Run `make check` before handing off broad or public API changes.

When touching voice/session behavior:

- Start with tests around `tests/test_agent_session.py`, `tests/test_room_io.py`,
  `tests/test_endpointing.py`, `tests/test_interruption/`, `tests/test_tools.py`,
  and the fake model files in `tests/fake_*.py`.
- Be careful with cancellation and async task ownership. Many components rely on
  explicit `aclose()`, channel closure, task cancellation, and event ordering.
- Watch for realtime-model differences. Realtime models can provide server-side
  turn detection and audio output, so they do not always follow the STT -> LLM ->
  TTS path.
- Keep speech/interruption changes deterministic enough for unit tests.

When touching worker/job/IPC behavior:

- Read `worker.py`, `ipc/proc_pool.py`, `ipc/job_proc_executor.py`,
  `ipc/job_thread_executor.py`, and `ipc/job_proc_lazy_main.py` together.
- Consider process and thread executors. Windows defaults to thread execution;
  other platforms default to processes.
- Preserve job status reporting, shutdown acknowledgements, drain timeout,
  memory warning/limit behavior, and load reservation semantics.
- Test with `tests/test_ipc.py`, `tests/test_drain_timeout.py`, and any affected
  worker/session tests.

When adding or changing a provider plugin:

- Match nearby provider patterns before inventing a new shape.
- Implement the provider-neutral interface: `llm.LLM`, `llm.RealtimeModel`,
  `stt.STT`, `tts.TTS`, `vad.VAD`, or the relevant avatar/workflow interface.
- Provide `model` and `provider` properties when possible so telemetry is useful.
- Emit metrics and provider request IDs where existing base classes support them.
- Register the plugin in `__init__.py` with `Plugin.register_plugin(...)`.
- Add `py.typed`, exports, version, README, and dependency metadata.
- Add unit tests with fakes where possible. Provider live tests usually require
  secrets and may run only in CI.
- If the plugin needs local model files, implement `download_files()` and verify
  `python agent.py download-files` or the plugin-specific equivalent.

When touching tests:

- Prefer deterministic fake STT/TTS/LLM/VAD implementations from `tests/fake_*.py`.
- Keep provider-key tests isolated. CI has secret-gated plugin jobs; forked PRs
  may not run them.
- `pytest` ignores `examples/` by default through `pyproject.toml`.
- Some CI paths download NLTK `punkt_tab`; local tokenizer tests may need the
  same data if they exercise NLTK-backed code.

## Quality Gates

Local minimum for small, focused changes:

```bash
uv run pytest path/to/test_file.py -k relevant_test
uv run ruff check path/to/changed_package
uv run ruff format --check path/to/changed_package
```

Recommended before a PR:

```bash
make fix
make check
make unit-tests
```

CI additionally runs package builds, type checks on Python 3.10 and 3.13, unit
tests, selected plugin/provider tests, evaluation tests, and native wheel builds
for packages such as `livekit-blingfire`, `livekit-blockguard`, and
`livekit-durable`.

## Local RTC SDK Development

The package dependency named `livekit` is the Python RTC SDK. If a change needs
local RTC SDK work, this repo expects a sibling `../python-sdks/livekit-rtc`.

```bash
make doctor
make link-rtc
make link-rtc-local
make unlink-rtc
make status
```

Do not use these targets unless the change actually needs local RTC/FFI work.

## Environment Variables

- `LIVEKIT_URL`: WebSocket URL of LiveKit server.
- `LIVEKIT_API_KEY`: API key for authentication.
- `LIVEKIT_API_SECRET`: API secret for authentication.
- `LIVEKIT_AGENT_NAME`: agent name for explicit dispatch (optional).
- Provider-specific keys: `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`,
  `ANTHROPIC_API_KEY`, and similar names used by each plugin.

## Common Risk Areas

- Public API churn around `AgentSession`, `Agent`, `WorkerOptions`, `JobContext`,
  tools, and turn-handling options.
- Async cancellation leaks in voice pipeline, room I/O, IPC, or provider streams.
- Process/thread executor divergence.
- Realtime model behavior that bypasses STT/TTS assumptions.
- Tool execution loops and max-tool-step behavior.
- Interruption, endpointing, and false-interruption recovery.
- Provider API drift, especially model names, request formats, streaming events,
  and retry/error semantics.
- Telemetry and metrics regressions caused by missing `model`, `provider`,
  request IDs, or span attributes.

When in doubt, trace the smallest runtime path end to end, then add a narrow test
at the boundary where the behavior changed.
