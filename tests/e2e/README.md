# E2E journeys

Real processes, sockets and services that fake ports and the DST cannot model. This folder is
collected only with `--e2e`:

```sh
uv run --no-sync pytest tests/e2e --e2e
uv run --no-sync --env-file project.env pytest tests/e2e --e2e   # with the live journeys
```

Each test states its prerequisites, actions, expected outcomes and bounds
(`docs/internals/testing.md`, "E2E contracts"). Business effects are checked through recorded
service calls or authoritative state, delivery through what the caller observed; never through
logs or model text. There are no retries; every action is bounded.

The `e2e` fixture runs ordinary application modules from `apps/` and owns their processes,
text clients and RTC callers. Correctness apps use scripted models and deterministic speech;
application logic, native execution, storage and transports remain real. Restart preserves
the app's data directory and creates a fresh process.

```python
async def test_a_spoken_reply(e2e):
    app = await e2e.start_app("tests.e2e.apps.voice_conversation", voice=True)
    async with e2e.voice_client(app) as caller:
        await caller.say("what time is it")
        reply = await caller.hear()
        assert reply.text == "it is noon"
        assert reply.audio_seconds > 0
```

Use `e2e.text_client(app)` for an owned public AgentClient, `app.restart()` for a graceful
restart and `app.kill()` followed by `app.start()` for a crash. Apps export their ordinary
AgentServer as `server`. A text app serves local HTTP without registration; a voice app joins
the isolated SFU. Keep those apps separate when the local SFU has no HTTP-door registration.
An external ASGI application exports `create_app(url)`; the runner supplies its bound base URL
and serves it in a process that imports no LiveKit SDK. `apps/stock_a2a.py` uses the upstream
A2A SDK's executor, task store and HTTP/SSE routes. `test_stock_a2a.py` delegates to it through
an ordinary RemoteAgent tool and resumes its input-required task after frontend restart.
`apps/adk_peer.py` uses ADK's ordinary agent, tool and
[A2A server API](https://adk.dev/a2a/quickstart-exposing/) with scripted inference. Pass
`python=` to `start_app()` when the peer needs its own dependency environment.
The shared `framework/media.py` also supplies the v1 compatibility speech fixtures.
See the [framework design](../../docs/proposals/e2e.md) for provider and acceptance policies.

## Live conversation evaluations

Real models are an explicit additional run:

```sh
uv run --no-sync pytest tests/e2e/test_delegation_live_models.py --e2e --live-models
```

Configure the cloud project's `LIVEKIT_URL` and its `LIVEKIT_INFERENCE_API_KEY` and
`LIVEKIT_INFERENCE_API_SECRET` (or `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`). The selected
project must match the inference gateway. `E2E_FRONTEND_MODEL` and `E2E_ROUTING_MODEL` select
the two model IDs; both default to `google/gemma-4-31b-it`, using separate model instances.
There is no scripted fallback when a selected model or its credentials fail.

The journeys use a real frontend, a real routing model and independent stock A2A services.
They check date responses, side questions, new topics, ambiguity and grouped name collection
with full and partial answers. Each writes `live-model-report.json` beneath its pytest artifact
directory: model IDs, turn timings, perceived conversation, original communication events and
task/input state. Assertions check original-task routing and the person's exact forwarded words;
they allow the model to choose natural wording. Keep failed runs when evaluating model quality.

## Files

| File | Journey | Live or scripted |
| --- | --- | --- |
| `test_framework.py` | Conversation restart, received speech and interruption during actual audio playback | Scripted models and tone speech; real app processes, SQLite and isolated SFU |
| `test_stock_a2a.py` | Remote Message replies and input-required continuation after frontend restart | A stock A2A SDK server in a separate process; ordinary frontend, scripted model, real HTTP/SSE and SQLite |
| `test_adk.py` | The same RemoteAgent frontend runs an ADK agent and its tool | Pinned ADK and A2A SDK in a separate environment/process; scripted inference, real server and HTTP transport |
| `test_delegation_live_models.py` | Intent routing, grouped remote questions and partial responses | Opt-in real frontend and routing models; stock A2A peer processes and real HTTP/SSE |
| `test_worker.py` | Job processes: IPC launch, health kill, wire faults, stress; a real worker's crashed, leaking, slow-initializing and orphaned jobs and their logs; a process that dies holding a storage turn; worker processes serving their text doors | Hermetic: real OS processes, local SQLite (AgentDB case skips without a project) |
| `test_cli.py` | The CLI launcher: module launch, failure exits, one shutdown budget across drain and cleanup, signals while a real job process runs | Hermetic: real OS processes and signals |
| `test_preload.py` | Application preloads before readiness, spawn and forkserver, copy-on-write, unsafe preloads refused | Hermetic; forkserver cases run on Linux only |
| `test_text.py` | Text across two workers: shared history and idempotent replay, input required and approval answered on the other worker, a worker killed mid-turn; a dropped stream read back by `tasks/get` | Worker processes need live AgentDB; in-process servers run on SQLite (and AgentDB when configured) |
| `test_endpoints.py` | HTTP and WebSocket through the LiveKit node: streaming, drain, abnormal close, token forwarding; a console worker delegating to its own endpoint over loopback | Node journeys need a live project; the console journey is hermetic |
| `test_room.py` | Room waits settle on a real SFU disconnect; a later publication is seen | An isolated `livekit-server` (`sfu.py`) |
| `test_job_regressions.py` | Past worker bugs through real job processes: tool subprocesses, crashed entrypoints, child log levels, explicit credentials, dev vs production drain, failed spawns | Hermetic: real OS processes, local aiohttp stand-ins |
| `test_room_regressions.py` | Past room bugs: participant waits, concurrent connects, room deletion, agent leaving, dispatch metadata, background and initial speech, encrypted data, callback publication, native memory | An isolated `livekit-server` (`sfu.py`) |
| `test_v1_client_compat.py` | v1 clients (client-sdk-js, components-js/agents-ui, Swift, Android, Flutter, React Native, the starters, agents-playground, ESP32) finding, hearing, dispatching, calling and losing a 2.0 agent; the playground's `lk.agent.session` protocol; avatar workers | An isolated `livekit-server` (`sfu.py`); a real worker and job, scripted models (`v1_clients.py`) |
| `test_v1_client_compat_conversation.py` | The same v1 clients' chat, transcription streams, legacy packets, interruptions, expressions and pre-connect audio per SDK family | An isolated `livekit-server` (`sfu.py`); a real worker and job, scripted models (`v1_clients.py`) |
| `test_v1_client_compat_avatar.py` | Avatars: the person hears and sees the avatar speak a reply, a cut-in clears it, legacy packets name its microphone; a v1 avatar worker against a 2.0 agent and a 2.0 avatar worker against a v1 agent, both ways over the plain `lk.playback_finished` | An isolated `livekit-server` (`sfu.py`); a real job, the real `DataStreamAudioReceiver` and `AvatarRunner` with `FakeVideoGenerator`, v1 peers in `v1_clients.py` |
| `test_room_io_regressions.py` | Room text in and transcription out across sessions, encrypted microphone audio, an unanswered room deletion at shutdown; a Python stage over the avatar data-stream output (playback reports, clear, a late avatar) | An isolated `livekit-server` (`sfu.py`); the deletion case is hermetic |
| `test_audio_processors.py` | Python audio frame processors on room input: what the STT hears, track replacement, failures | An isolated `livekit-server` (`sfu.py`) |
| `test_noise_cancellation_compat.py` | Shipped noise cancellation models (NC, BVC, Krisp, ai-coustics) passed as `AudioInputOptions(noise_cancellation=...)` | An isolated `livekit-server`; the Cloud case needs a LiveKit Cloud project |
| `test_encryption.py` | End-to-end encrypted rooms: media and data between a job and a person | An isolated `livekit-server` (`sfu.py`) |
| `test_rpc_regressions.py` | RPC to a missing method under short deadlines | An isolated `livekit-server` (`sfu.py`) |
| `test_simulation_regressions.py` | A simulator interrupts a greeting in a simulated job | An isolated `livekit-server` (`sfu.py`), a real job process |

Only the models (`tests/fakes.py`) and the application service (`resources.RecordingService`)
are scripted; agent execution, tools, approval, leases, persistence and transports are the
production implementation.

The lean text storage journey runs against two loopback servers sharing SQLite, and against
AgentDB when configured. It checks `bind()` checkpoints and an unsaved control, reading
persisted conversation state through fresh requests on the other server. Application fields
still require `save_context`:

```sh
uv run --no-sync pytest tests/e2e/test_text.py --e2e -k text_session_storage
```

## Prerequisites

- ADK interoperability: install its pinned dependencies in an isolated environment and select
  that interpreter. CI prepares it automatically; locally:

  ```sh
  uv venv --python .venv/bin/python /tmp/agents-adk-peer
  uv pip install --python /tmp/agents-adk-peer/bin/python -r tests/e2e/adk.requirements.txt
  ADK_E2E_PYTHON=/tmp/agents-adk-peer/bin/python uv run --no-sync pytest tests/e2e/test_adk.py --e2e
  ```

- Live journeys: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` naming a project with
  AgentDB. A missing variable fails the selected test instead of skipping it. Against a local
  development node, also set `LIVEKIT_WORKER_INSECURE=1` for `test_endpoints.py`.
- Room journeys: `livekit-server` on `PATH`; `sfu.py` runs its own on free ports with private
  keys, so a developer's server on :7880 is never touched.
- Noise cancellation: `livekit-plugins-noise-cancellation` and `livekit-plugins-ai-coustics` are
  dev dependencies; the LiveKit Cloud case skips without a project.

## Layout

- `conftest.py`: fixtures owning worker processes, threads, storages and job runners. Cleanup runs on failure and cancellation and reaps every child.
- `resources.py`: process, in-process server and recording-service ownership, and the bounds.
- `drivers.py`: A2A actions and wire observations; no verdicts.
- `sfu.py`: the isolated SFU fixture and the room helpers (tokens, connections and tones).
- `ready.py`: atomic process-readiness and receiver-observation files.
- `v1_clients.py`: the agent the v1 client compatibility journeys dispatch (tone STT and TTS),
  and the client moves they repeat, each as the v1 client does it on the wire.
- `processes.py`: process-tree evidence captured before cleanup, and the job-runner builders.
- `text_worker.py`, `job_worker.py`, `forkserver_freeze.py`: the applications the tests run as
  separate processes.
