# Trace agent startup before changing publication

This example starts a real named `AgentServer`, uses a browser button to join a room with an agent dispatch in its token, and runs a real `AgentSession`. It measures the path to non-silent audio in a browser AudioWorklet. It changes no SDK implementation. Instrumentation wraps existing startup and output methods and collects the SDK's OpenTelemetry spans.

[Native preparation fix and latest staging results](native-preparation-reports/README.md) · [Actual pre-PR comparison](before-after-reports/README.md)

[Measured staging results and limitations](reports/README.md) · [Phase timelines](reports/timeline.html)

The profiles use existing APIs:

- `example-order`: follows the connection/output ordering in the local `myAgent_perplexity.py`: connect and find the participant first, construct/start the session, then start a custom audio output. The name of that local file is historical; it actually uses OpenAI Realtime. The reproduction isolates startup ordering and omits that app's VAD/video/tools.
- `parallel-output`: still connects first, but starts the session and its custom output concurrently.
- `automatic`: lets `AgentSession.start()` connect and lets default RoomIO initialize output concurrently with activity startup.

The controlled cases are a fast synthetic greeting, an explicit 1.2-second asynchronous activity wait, and an explicit 1.2-second wait before generating audio. They use actual dispatch, native Python RTC, Cloud SFU transport, browser decoding and rendering. They do not measure a production model. The optional `realtime` mode calls OpenAI Realtime with a fixed short greeting.

## Run

Use an isolated environment from the repository root (the existing checkout environment may have older dependencies):

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e ./livekit-agents \
  -e ./livekit-plugins/livekit-plugins-openai playwright python-dotenv
.venv/bin/python -m playwright install ffmpeg
```

Google Chrome and Node must be installed. `LIVEKIT_JS_SDK_DIR` should point to a JS SDK checkout with its dependencies installed; it defaults to `~/dev/client-sdk-js` in this machine's example. Ports 18790, 18791, and 18793 must be free.

Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` for a test project. The runner creates uniquely named rooms and a uniquely named worker, then deletes its rooms and stops its processes. It does not stop any existing server or agent.

```sh
.venv/bin/python examples/startup_tracing/run.py \
  --mode controlled --profile all --repetitions 5 \
  --output /tmp/agent-startup-controlled

# The provider env file is read only for OpenAI variables; it cannot override the test project.
.venv/bin/python examples/startup_tracing/run.py \
  --mode realtime --profile both --repetitions 5 --provider-env /path/to/provider.env \
  --output /tmp/agent-startup-realtime

.venv/bin/python examples/startup_tracing/analyze.py \
  /tmp/agent-startup-controlled --output /tmp/agent-startup-report
```

Each case has an excluded warmup per profile. Execution order rotates across profiles. `--case controlled-activity-1200` narrows a run. The worker is configured with one idle process; traces record whether job assignment actually used a warm process.

## What is measured

- Client click, rendered connecting/connected/agent-state feedback, room connection, subscription, and first non-silent audio in the browser rendering graph.
- Worker job received/accepted/assigned, entrypoint, room connection, model construction, activity/toolset setup, `on_enter`, output publication, subscription, first PCM at the output, and first frame accepted by output.
- Existing OpenTelemetry spans, including observed event-loop stalls. PCM-ready means it reached `_ParticipantAudioOutput.capture_frame`; the controlled source also records its first generated frame.

The audio probe is prepared before the start action so its own initialization does not block connection. The browser UI is already loaded; page download, production authentication, microphone capture, STT/user turn detection, and physical speaker latency are excluded. Audio input is disabled to isolate first greeting. Browser recording uses muted HTML audio to activate Chromium's WebRTC playout, and a live WebAudio graph measures non-silent rendering. Headless Chrome mutes physical output. Successful samples include an audio recording, screenshot, and selected UI screen recordings.

Both the worker and browser run on this host, allowing their epoch timestamps to be aligned; the SFU is remote. No artificial signaling or media delay is applied. Recordings contain only the synthetic greeting or generated fixed greeting.

## How to interpret the report

The report compares first PCM availability with media readiness. If subscription is ready before PCM, publication has slack and shortening it alone does not improve first audio. If PCM waits for media, some media setup is exposed on the critical path.

The `instant_registration_saving_bound_ms` column is **not a measured optimization**. It is a simple counterfactual bound: `min(publication duration, max(media ready - PCM ready, 0))`. The analogous post-JOIN bound includes publication plus subscription. It assumes shifting those phases earlier with everything else fixed, so it is not a prediction of a Rust change or of remote subscriber scheduling.

Use `timeline.html` for aligned phase bars, or load the per-run `.trace.json` files into a trace viewer. Raw client results and agent spans remain in the run directory. Synthetic delays and failed provider attempts must be reported separately from real-provider results. SDK/package and browser versions are captured in each run's manifest.
