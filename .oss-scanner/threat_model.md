# LiveKit Agents threat model

LiveKit Agents is a Python framework for realtime voice and multimodal AI
agents. An agent server (worker) registers with a LiveKit server over a
websocket, accepts job assignments, and runs each job in a child process that
joins a LiveKit room. Inside the job, an `AgentSession` turns participant
audio, text, and data into STT, LLM, and TTS calls through provider plugins,
runs the developer's function tools, and publishes audio and transcripts back
to the room.

## Scope and trust boundaries

Audit the hand-written Python on `main`. The repository has 80+ plugins, too many to cover evenly, so focus on the `livekit-agents` core and
the representative plugins listed below. Prioritise:

- `livekit-agents/livekit/agents/voice/room_io/`: participant audio, chat text
  streams (`lk.chat`), pre-connect audio byte streams, linked-participant
  selection, and `publish_on_behalf` attribute handling.
- `voice/remote_session.py`: the session-host protocol (`SessionHost`,
  `RoomSessionTransport`, `TcpSessionTransport`). Room participants send it
  protobuf requests over byte streams: `run_input`, `get_chat_history`,
  `get_agent_info`, `update_io`, `finalize_simulation`, and others.
- `voice/avatar/_datastream_io.py`: avatar audio byte streams and the
  playback/clear-buffer RPCs, which filter on caller identity.
- `llm/utils.py`, `llm/tool_context.py`, `voice/tool_executor.py`,
  `voice/generation.py`, `llm/_provider_format/`: parsing LLM tool-call
  arguments (strict JSON, then `json_repair`, then pydantic validation) and
  binding them to tool functions. `llm/mcp.py`: MCP tool results.
- `worker.py`: the worker protocol (`register`, `availability`, `assignment`,
  `termination`), job-token handling, and the HTTP servers. The health server
  (`/`, `/worker`) binds to all interfaces without authentication, on port
  8081 in production mode. The optional Prometheus server (`/metrics`) uses the same
  host.
- `ipc/`: the protobuf channel between the worker and job processes, and the
  pickled log queue (`ipc/log_queue.py`). The worker and its job processes are
  one trust domain, since both run developer code. A finding there needs
  participant- or provider-controlled bytes to reach the unpickler, not just
  strings inside log records.
- `utils/codecs/decoder.py`, `utils/audio.py`, `utils/images/`: decoding audio
  and images, including opus from pre-connect byte streams.
- `telemetry/` and `cli/log.py`: redaction of credentials and PII in logs and
  traces.
- Representative plugins: `livekit-plugins-openai` (LLM, realtime, tools),
  `livekit-plugins-deepgram` (STT websocket), `livekit-plugins-cartesia` (TTS
  websocket), `livekit-plugins-silero` (local ONNX VAD),
  `livekit-plugins-turn-detector` (model download and local inference), and
  `livekit-plugins-browser`, whose `browser/navigate`, `go-back`, and
  `go-forward` RPCs and `browser-input` data packets let room participants
  drive a Chromium page. Navigating to a participant-chosen http(s) URL is the
  feature. Non-http(s) schemes, internal addresses, and navigation by a
  participant who doesn't hold focus are in scope.

Out of scope: `examples/`, `tests/`, `scripts/`, `.github/`, generated
protobuf code, and plugins excluded from the workspace
(`livekit-plugins/community`, `livekit-blockguard`, `livekit-durable`).
Report a bug in another plugin when it matches a pattern listed here.

Untrusted input:

- Anything another room participant controls: audio and video tracks, data
  packets, text and byte streams, RPC invocations and payloads, and
  participant identity, attributes, and metadata.
- User speech and text that reach LLM prompts, and therefore the tool-call
  names and arguments an LLM emits.
- Job metadata and dispatch fields the server relays. The server is trusted,
  but these values can originate from an end user's token or a dispatch
  request.

Semi-trusted: responses from third-party STT, TTS, LLM, avatar, and MCP
providers. Treat them as authentic but possibly malformed. A malformed
response may fail the request, but it must not crash the worker or corrupt
another session.

Trusted: the agent developer's code and configuration, including tool
definitions, instructions, and plugin options. Also trusted: API keys and
secrets, the LiveKit server and the worker websocket (authenticated with the
API key and secret; the job assignment token is a signed JWT), and the local
machine and filesystem.

LiveKit server (`livekit/livekit`) is enrolled separately. Report bugs in how
the server authorises participants there. Report bugs in how this framework
handles what participants send here.

## Security properties

- **Participant isolation:** input from a participant that the session isn't
  linked to must not drive the session. This covers audio, chat text,
  pre-connect audio for another participant's track, avatar RPCs from an
  unexpected caller, and session-host requests when `session_host=False` or a
  `remote_identity` is set.
- **Session host:** when `session_host` is on (the default), any room
  participant can observe and drive the session. That is documented
  behaviour. Requests must not reach more than their documented effect, for
  example by leaking secrets or API keys or reading another session's state.
- **Tool boundaries:** the LLM can call only the tools registered on the
  current agent, with arguments that pass the tool's schema. Repaired or
  malformed arguments must not bypass validation, select an unregistered
  tool, or inject a `RunContext` or other framework-only parameter.
- **Parsing:** protobuf, JSON, audio, and image input from participants or
  providers must not crash the worker process, hang the event loop, or
  allocate without bound. Watch byte streams that are buffered in full before
  parsing.
- **Worker:** a job's crash or hang must not take down the worker or other
  jobs. The health and metrics endpoints must not expose secrets or allow any
  state change.
- **Secrets:** API keys, API secrets, worker tokens, and provider credentials
  must not appear in logs, traces, exception messages, health or metrics
  output, or data sent to the room. With redaction enabled, user PII must not
  reach telemetry. The tests `test_ws_handshake_credential_redaction.py`,
  `test_telemetry_pii_redaction.py`, and `test_toolerror_log_pii.py` encode
  part of this.
- **Files and URLs:** participant-supplied values must not choose local file
  paths or make the agent fetch local files or internal addresses.

## Severity guidance

Prompt injection is inherent to LLMs. "The LLM can be talked into calling a
tool or saying something" is not a finding. It becomes one when the framework
itself mishandles the boundary: running a tool that isn't registered, passing
unvalidated arguments, leaking secrets into the prompt or the room, or
treating LLM output as trusted code or paths.

- **Critical:** remote code execution or arbitrary file read/write in the
  worker or a job process from a room participant or provider response.
- **High:** a room participant drives or reads a session the framework says it
  can't (an identity-filter bypass); tool-call handling that runs an
  unregistered tool or skips argument validation; API secrets or provider keys
  leaked to the room, logs, or health output; a crash or hang of the worker
  process (not just one job) reachable from a participant or provider; SSRF to
  internal addresses or local files from participant input.
- **Medium:** a crash, hang, or unbounded memory in one job process from a
  participant or a malformed provider response; PII that escapes redaction
  while redaction is on; unauthenticated health endpoints exposing more than
  status and worker info.
- **Low / not a finding:** prompt injection that stays within the registered
  tools and their schemas; information the session host returns to
  participants by design; issues that need the developer's code, config, or
  secrets; malformed messages only the LiveKit server can send; examples and
  tests; provider-side bugs.

## Testing

The checkout is at `/src`, with its virtualenv at `/src/.venv` (on `PATH`).
`uv sync --all-extras --dev` ran during the build. `UV_OFFLINE=1` and
`UV_NO_SYNC=1` keep `uv run` from downloading or re-syncing. NLTK `punkt_tab`
is in `/usr/share/nltk_data`, and the `o200k_base` and `cl100k_base` tiktoken
encodings are in `/usr/share/tiktoken`. No provider API keys are present. Tests marked
`plugin`, `stt`, `tts`, `realtime`, or `evals` need provider keys or network
access and won't run here.

```sh
cd /src
# the hermetic CI gate; test_room.py needs a LiveKit server started via docker
uv run --no-sync pytest --unit --audio_eot --ignore=tests/test_room.py
uv run --no-sync pytest --list-categories
uv run --no-sync pytest tests/test_tools.py tests/test_llm_utils.py
uv run --no-sync pytest tests/test_remote_session.py tests/test_session_host.py tests/test_room_io.py
```

Every test module needs a category marker (`pytestmark = pytest.mark.unit`)
or collection fails. Pass `--allow-uncategorized` for a scratch reproducer.
The fakes in `tests/` (`fake_llm.py`, `fake_stt.py`, `fake_tts.py`,
`fake_session.py`, `fake_io.py`) drive an `AgentSession` without providers.

The build runs `git lfs pull`. If that failed, the media and model files in
`livekit-agents/livekit/agents/resources/`, `tests/`, and the silero plugin
are LFS pointer files. None of the unit tests read them.

## Reports

State the attacker's position: room participant (and with which
permissions), provider response, LLM output from injected input, or server.
Give the entry point (a topic, RPC method, or message type) and the
demonstrated impact. Provide a minimal reproducer, preferably a pytest module
under `tests/` that uses the existing fakes.
