# E2E journeys

Real processes, sockets and services that fake ports and the DST cannot model. This folder is
collected only with `--e2e`:

```sh
uv run --no-sync pytest tests_new/e2e --e2e
uv run --no-sync --env-file project.env pytest tests_new/e2e --e2e   # with the live journeys
```

Each test states its prerequisites, actions, expected outcomes and bounds
(`docs/internals/testing.md`, "E2E contracts"). Business effects are checked through recorded
service calls or authoritative state, delivery through what the caller observed; never through
logs or model text. There are no retries; every action is bounded.

## Files

| File | Journey | Live or scripted |
| --- | --- | --- |
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

Only the models (`tests_new/fakes.py`) and the application service (`resources.RecordingService`)
are scripted; agent execution, tools, approval, leases, persistence and transports are the
production implementation.

## Prerequisites

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
- `sfu.py`: the isolated SFU fixture and the room helpers (tokens, connections, tones, markers).
- `v1_clients.py`: the agent the v1 client compatibility journeys dispatch (tone STT and TTS),
  and the client moves they repeat, each as the v1 client does it on the wire.
- `processes.py`: process-tree evidence captured before cleanup, and the job-runner builders.
- `text_worker.py`, `job_worker.py`, `forkserver_freeze.py`: the applications the tests run as
  separate processes.
