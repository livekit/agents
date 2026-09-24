# Microsoft AI speech plugin for LiveKit Agents

**STT and Azure Speech TTS have bounded live smoke coverage.** The TTS path has
been verified with MAI-Voice-2-Flash (Harper, PCM16 mono at 24 kHz), including
playback through a local LiveKit room using the installed plugin wheel and
released `livekit-agents==1.8.2`.

The current package follows the synchronized 1.8.3 release and requires
`livekit-agents>=1.8.3`. The live results below describe the earlier 1.8.2
validation; the updated wheel and examples are checked offline against released
1.8.3 without claiming an additional live run.

The streaming STT path has separately transcribed one short synthetic English
utterance through the installed wheel, using the Azure GA transcription route,
explicit `api-key` authentication, PCM16 mono at 16 kHz, and a client commit.
The acknowledged final matched every expected word, including the last word,
without added silence or promoting an interim hypothesis. The endpoint
acknowledged the configured 16 kHz rate before any audio was sent.

A separate five-turn synthetic browser/WebRTC test passed through a local
LiveKit server, real MAI STT with its own local VAD, the model-less echo example,
and real MAI TTS back to browser audio. All five STT finals matched the complete
expected words without duplicates. The test covered a brief internal pause,
barge-in that cleared the old echo without stale output, explicit
disconnect/reconnect, and closure of both sessions and providers. The fourth
echo was intentionally interrupted; the other echoes completed. The microphone
track remained open with ordinary inter-turn silence; no extra tail padding
or manual per-utterance commits were used.

This does **not** establish access in every resource/region, recognition
accuracy across inputs/languages, every backend tail boundary, long-session
reliability, physical microphone behavior, or subjective voice quality.
Hermetic tests also cover the client lifecycle and VAD ordering. None of these
results is a model-latency benchmark. An Azure Speech TTS resource/key does
**not** establish access to the separate STT service.

There is no LLM, speech-to-speech realtime model, Azure OpenAI convenience
constructor, provider catalog, token minting, or OpenAI credential/model default.

The implementation adapts the Apache-2.0-licensed LiveKit OpenAI plugin's package
and STT/TTS interfaces, and the Azure plugin's REST transport pattern. It does
not depend on either plugin. Transport, transcription finalization, and the
escaped SSML/WAV mapping are specific to this integration. See `LICENSE` and `NOTICE`.

## Local installation

From this repository, install the workspace package rather than assuming a
published distribution exists:

```sh
uv sync --package livekit-plugins-microsoft-ai --no-default-groups
```

## Configuration

Pass constructor arguments or set the following environment variables. There
are deliberately no default endpoints, models, voice IDs, or TTS sample rate.
Obtain these values and the exact contract from your deployment owner.

| Environment variable | Constructor argument |
| --- | --- |
| `MICROSOFT_AI_STT_URL` | `STT(url=...)` |
| `MICROSOFT_AI_STT_API_KEY` | `STT(api_key=...)` |
| `MICROSOFT_AI_STT_AUTH_HEADER` | `STT(auth_header=...)`: `Authorization` or `api-key` |
| `MICROSOFT_AI_STT_MODEL` | `STT(model=...)` |
| `MICROSOFT_AI_STT_LANGUAGE` | `STT(language=...)` (optional) |
| `MICROSOFT_AI_TTS_URL` | `TTS(url=...)` |
| `MICROSOFT_AI_TTS_REGION` | `TTS(region=...)` (when no URL is configured) |
| `MICROSOFT_AI_TTS_API_KEY` | `TTS(api_key=...)` |
| `MICROSOFT_AI_TTS_MODEL` | `TTS(model=...)` |
| `MICROSOFT_AI_TTS_VOICE` | `TTS(voice=...)` |
| `MICROSOFT_AI_TTS_SAMPLE_RATE` | `TTS(sample_rate=...)` |
| `MICROSOFT_AI_ENV_FILE` | `STT(env_file=...)`, `TTS(env_file=...)` |

URLs are complete endpoints, including any required path and query string. For
example, `wss://stt.example.invalid/v1/realtime?intent=transcription` is a **dummy**,
not a Microsoft service address. No path or model query parameter is appended.
TLS is required except for loopback development endpoints. For TTS, a configured
full URL wins over `region`, regardless of which configuration source provides
each. Only when the URL is unset does an explicitly supplied region select
the standard public-cloud endpoint. An explicit or configured empty/whitespace
URL is an error, not permission to fall back to a region. Other required empty
values also fail rather than falling back silently.

TTS sends the Azure Speech resource key as `Ocp-Apim-Subscription-Key`, **not**
as a raw-key Bearer token. STT preserves its `Authorization: Bearer ...` default.
For an Azure realtime endpoint using resource-key authentication, explicitly set
`MICROSOFT_AI_STT_AUTH_HEADER=api-key` (or `auth_header="api-key"`); it sends the
raw credential from `MICROSOFT_AI_STT_API_KEY` as the `api-key` header, with no
Bearer prefix. The selector accepts only the exact values `Authorization` and
`api-key`; an empty/unknown value is an error. No auth fallback or automatic
scheme detection occurs, and credentials are never added to the URL.

Explicit `headers` (including `{}`) override the STT selector and credential
environment settings and cannot be combined with `api_key` or `auth_header`
constructor arguments. Use them only for a confirmed alternate authentication
scheme. No credentials are read from OpenAI/Azure variables. Caller-supplied
`http_session` objects are borrowed; otherwise providers own lazy sessions and
close them in `aclose()`.

Keep actual connection information in your environment or a user-selected
local dotenv file **outside the checkout** (or a deliberately ignored file).
The providers load a file only when `env_file` or `MICROSOFT_AI_ENV_FILE` selects
one; there is no automatic `.env` discovery. Files are parsed with python-dotenv,
without shell sourcing, variable interpolation, environment mutation or value
logging. Precedence is constructor argument, then process environment, then the
selected file. An empty required value fails instead of falling back silently.
Use owner-only file permissions and never copy this file into commits.

The smoke script additionally accepts `--env-file`. It preflights all selected
service configurations before making any request, so an empty/partial template
cannot accidentally start a selected live test. On POSIX it requires the selected
file to have mode `0600`. Never commit endpoints,
credentials, recordings, transcripts or request/response captures. Provider
errors deliberately omit response bodies and transport exception details that
could echo this information.

## STT contract and lifecycle

For the Azure GA **transcription** endpoint, the official
[transcription example](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/realtime-audio-websockets#transcribe-audio-in-real-time)
uses `/openai/v1/realtime?intent=transcription`; the deployment name is sent in
`session.audio.input.transcription.model`, not added as a URL query parameter.
Configure the full URL in `MICROSOFT_AI_STT_URL` and the deployment in
`MICROSOFT_AI_STT_MODEL`. The plugin sends that URL unchanged; it does not add
the conversation API's `model=` query or preview `deployment`/`api-version`
parameters. Do not put a key in the URL. This routing/auth documentation alone
does not establish audio-rate or transcript-event compatibility for a new
deployment; validate the MAI contract below independently.

```python
from livekit.agents import inference
from livekit.plugins import microsoft_ai

detector = inference.VAD(model="silero")
speech_to_text = microsoft_ai.STT(vad=detector, language="en")
text_to_speech = microsoft_ai.TTS()
```

`vad` is explicit. Pass a LiveKit VAD with ordered `INFERENCE_DONE` events even
during silence, input-relative timestamps, and `START_OF_SPEECH.frames`
containing the detected onset and prefix through that timestamp (the bundled
Silero VAD provides these). Empty
or incompatible start frames fail explicitly rather than clipping the onset.
Alternatively, pass `vad=None` and call `flush()` / `end_input()` yourself.
**Configuring only AgentSession's VAD is insufficient:** it does not commit
native STT streams.

The client protocol is:

1. Await `session.created`, send `session.update`, await `session.updated`.
2. Configure a transcription session with `audio.input.format` equal to
   `{"type": "audio/pcm", "rate": 16000}`, a required transcription `model`,
   optional `language`, and `turn_detection: null`, `noise_reduction: null`.
3. Send base64 PCM16 little-endian mono audio as `input_audio_buffer.append`.
4. For an item identified by `item_id`, an
   `conversation.item.input_audio_transcription.intermediate` event's
   `intermediate` replaces the revisable hypothesis. A `.delta` event's `delta`
   appends finalized text and clears the hypothesis. Both produce LiveKit
   interim results, not final utterances.
5. Drain audio, send `input_audio_buffer.commit`, await
   `input_audio_buffer.committed` with `item_id`, then `.completed` with the
   authoritative `transcript`. Only this emits a LiveKit final transcript,
   followed by end-of-speech. The socket stays open for subsequent utterances.

New deployments must be verified to implement these event fields and
handshake/commit ordering. A short manual-commit smoke does not exercise every
interim revision or VAD boundary. HTTP statuses are preserved; WebSocket
`error` / transcription `.failed` events are terminal unless a pre-audio
transient status is supplied. The initial error mapping recognizes
`error.status_code`, `invalid_api_key`, `rate_limit_exceeded`, `content_filter`
and `safety_violation`; it never disables or works around safety checks.

VAD inference timestamps serialize audio and turn boundaries, so later audio
cannot overtake an earlier commit even if VAD processing or uploads are slow.
Mono input at other sample rates is resampled by the SDK. Stereo is rejected.
Transport frames are 50 ms; the final shorter frame and resampler/VAD remainder
are sent without rounding away samples or adding synthetic padding.

With VAD, idle inference windows are discarded locally, not uploaded to an
uncommitted server buffer. At speech start, the VAD's actual frames restore
the complete detected onset/prefix; no guessed pre-roll duration or private
VAD settings are used. Only overlap with a previously committed turn is removed.
The prefix is framed and flushed before subsequent audio so it is not counted
twice for backpressure. Speech and the VAD's observed end-of-speech silence are
uploaded in order, then committed; prolonged inter-turn silence sends neither
audio nor empty commits. No provider clear/keepalive events are invented.

**Tail limitation:** sending every byte and receiving `.completed` proves
transport completion, not that the backend decoded an incomplete model chunk.
There is no invented padding rule. An obviously discarded outstanding
hypothesis fails explicitly rather than being promoted to a fabricated final.
A live test must verify the full expected transcript, particularly the last
word, for both a short clip and a non-chunk-aligned tail. Obtain a documented
backend drain/flush mechanism if commit does not decode the tail.

After VAD detects speech, `flush()` drains its real audio tail and commits,
waiting internally before processing subsequent input; it leaves the socket
open. `end_input()` also waits for the acknowledged final and closes the socket.
Flushing or ending idle VAD input produces no empty turn. To send a finite clip
regardless of whether a VAD detects speech, use `vad=None`; that manual mode
continues forwarding all input audio and requires caller-managed commits.
`aclose()` cancels immediately without committing or exposing buffered events.
Batch `recognize()` is unsupported and `offline_recognize=False`.

`APIConnectOptions.timeout` bounds connection, handshake, writes and
finalization. `max_retry` is a finite connection-only retry budget: after any
audio is consumed, a disconnect/error is surfaced without replay or hidden
reconnection. Reopening the stream is the caller's decision. Input is bounded
by `max_buffered_audio` (default 5 seconds) and 1,024 queued entries; overflow
fails explicitly. Each VAD start prefix is separately capped by the same
duration and fails rather than being truncated if oversized. The adapter keeps
no additional idle history; the VAD owns its bounded onset/prefix buffer.
Idle samples count as processed, so ordinary silence does not consume the
queued-audio allowance indefinitely. These bounds cover client lag/prefix
retention, not the length of an active utterance at the provider. Pace
prerecorded input rather than enqueueing entire files. Idle gating and delayed
onset recovery are covered by hermetic tests, not an additional live-service
accuracy or tail guarantee.

## Azure Speech TTS contract

This implementation follows Microsoft's public
[MAI voice documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-voices)
and [Speech REST reference](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech).
It posts SSML to the **exact configured synthesis URL**, with
`Content-Type: application/ssml+xml`, `Ocp-Apim-Subscription-Key`,
`X-Microsoft-OutputFormat`, `User-Agent`, and `Accept: audio/wav`.

Supply the full synthesis endpoint, including its documented
`cognitiveservices/v1` path and any resource-specific routing. A generic Azure
resource URL or SDK endpoint may not be a usable REST synthesis URL. The plugin
does not append a path, infer a region, rewrite the host, or follow redirects.
Confirm that the provided URL is correct before live validation.

Alternatively, supply `region` / `MICROSOFT_AI_TTS_REGION` without a URL. It
constructs `https://<region>.tts.speech.microsoft.com/cognitiveservices/v1`,
following the standard public-cloud Azure Speech convention. This is not a
region-availability catalog or access guarantee. Sovereign clouds and
custom/private deployments require an explicit full URL. There is no automatic
region detection, failover or redirection to a different region.

The full `voice` ID selects the voice and model in SSML. `model` is required
metadata and is checked against the voice ID's suffix, case-insensitively.
For example, the public documentation pairs `mai-voice-2-flash` with
`en-US-Harper:MAI-Voice-2-Flash`; availability still depends on the resource and
region. **Voice-2.1-Flash and Voice-2-Flash are not treated as aliases.**
SSML language defaults to `en-US`; override the `language` constructor argument
for other locales.

Text and attributes are XML-escaped structurally. Input text is always plain
text, not caller-supplied SSML; it cannot inject `<audio>` or other markup. No
speaker tags, `input`/`prompt` JSON, separate JSON `voice` property, token minting,
voice cloning, safety overrides or quality-disable controls are included.

The response must be HTTP 200 with a WAV content type and a complete,
uncompressed PCM16 mono WAV at the configured rate. Supported documented WAV
rates are 8,000, 22,050, 24,000, 44,100 and 48,000 Hz; at 24 kHz the output header
is `riff-24khz-16bit-mono-pcm`. Empty/truncated audio, wrong rates/channels,
JSON/base64 envelopes, SSE, MP3 and raw PCM are rejected, not guessed.
Request construction and decoding are isolated in `tts.py`; unsupported direct
JSON/Foundry variants are not silently tried as fallbacks.

`TTS.synthesize()` returns a `ChunkedStream` of correctly framed audio after
the complete response has been validated. `streaming=False` is intentional.
AgentSession supplies its existing sentence `tts.StreamAdapter` for incremental
LLM text; there is no claim of native text/audio streaming. Cancellation closes
the request and stops output, including already buffered frames.

Client-side safeguards, not advertised provider limits, are configurable:
`max_text_length=4096`, `max_audio_bytes=10485760`, and
`request_timeout=30` seconds. HTTP errors and safety refusals remain errors.
Only transient failures retry, with no audio emitted from incomplete attempts.

## Examples and validation

### Microphone echo with STT and TTS, without an LLM

`examples/microsoft/microsoft_ai_echo.py` receives a browser microphone track
through a real LiveKit room, transcribes it with `microsoft_ai.STT`, then echoes
the completed user turn with `microsoft_ai.TTS`. It uses
`Agent.on_user_turn_completed` and `AgentSession.say()`; `StopResponse` suppresses
an additional model reply. There is no LLM, manual transcription injection,
per-utterance `flush()` call, or paid LiveKit Inference.

The same local Silero VAD model is passed to **both** STT and AgentSession;
each opens its own stream. The plugin's VAD drives provider commits after
0.5 seconds of observed silence. Session VAD detects barge-in after 0.2 seconds
of speech. Echoes are interruptible and never automatically resumed after a
false interruption; preemptive generation and provider retries are disabled.
Do not add backend padding or promote interim text to a final to hide tail loss.

With the same local-server `LIVEKIT_*` settings described below and an external
file containing both providers' configuration:

```sh
uv run --package livekit-plugins-microsoft-ai --no-default-groups \
  python examples/microsoft/microsoft_ai_echo.py dev --no-reload --log-level info
```

In the browser, explicitly click Start microphone and grant permission.
Publish only a microphone audio track using the official `livekit-client` SDK
and a short-lived, room-scoped token minted server-side. Set `canPublish: true`
**and** `canPublishSources: ["microphone"]`; the browser SDK requires the
publish gate as well as the source allowlist. Do not grant video, room admin,
or remote agent control. Enable
playback from a user gesture, attach the agent audio, and consume the standard
`lk.transcription` streams for transient interim/final captions. No microphone
may start on page load or automatically after reconnect. Stop must release the
capture track and disconnect. Use headphones to avoid feeding the echo back in.

While connected, microphone audio is sent to the configured STT service and
recognized final text is sent to TTS. The example disables recording and remote
session control, accepts no typed-text input, and does not log transcripts.
Its room captions and session state are transient. Do not enable audio dumps,
debug transcript logs, external telemetry exporters, or browser recording when
testing private speech. Each room session is limited to three minutes and
closes both providers when its participant leaves. Initial participant/audio
readiness is bounded to ten seconds, so a failed browser join cannot leave
the example waiting for a user turn until its session limit.

Automated fixture audio published through the same browser/LiveKit track is a
useful transport and lifecycle test, but it does **not** validate a physical
microphone, acoustic echo cancellation, or subjective sound quality.

The bounded acceptance used the installed wheel with released Agents 1.8.2,
local LiveKit server 1.13.7 and browser client 2.22.3. It covered five short
synthetic turns across two STT sessions, including a 180 ms internal pause
below the configured 500 ms VAD silence threshold. The fifth turn interrupted
the fourth echo; the old speech handle was interrupted, its output cleared,
no server frames followed its completion, and browser audio energy stayed
flat during the observed quiet interval after transport settling. All five
STT finals, the four completed echoes, and disconnect cleanup passed.
This test did not count punctuation-only shutdown fallbacks as user turns.
A nonfatal SDK FFI-handle warning occurred during teardown; public session
and provider closure checks passed.

### TTS only in a real room, without a LiveKit Cloud account

`examples/microsoft/microsoft_ai_tts_room.py` uses only `microsoft_ai.TTS` and
`AgentSession.say()`. There is no LLM, STT, VAD, microphone, text-input handler,
remote session control or paid LiveKit Inference. It waits for a participant
and an audio subscription, says `Hello, this is a Microsoft AI voice test.`
once, waits for playback, closes its session and TTS client, and ends the
one-shot job. Synthesis retries and recording are disabled. A fresh room/job
triggers another paid TTS request; do not repeatedly reconnect to test playback
controls.

Use a [local open-source LiveKit server](https://docs.livekit.io/home/self-hosting/local/)
bound to loopback, or an existing LiveKit server. Set standard `LIVEKIT_*`
variables separately from the private Microsoft AI configuration:

```sh
export LIVEKIT_URL=ws://127.0.0.1:7880
export LIVEKIT_API_KEY=devkey
export LIVEKIT_API_SECRET=secret
export MICROSOFT_AI_ENV_FILE=/path/outside/checkout/endpoints.env

uv run --package livekit-plugins-microsoft-ai --no-default-groups \
  python examples/microsoft/microsoft_ai_tts_room.py dev --no-reload
```

`devkey` / `secret` are the public local-server development defaults, **not**
Azure credentials and not suitable for a production or externally exposed
server. Only the agent loads the private TTS file.

Connect a subscribe-only participant using the official `livekit-client`
browser SDK and a short-lived room-scoped token minted by a local backend.
Call `room.startAudio()` from a click, attach subscribed audio tracks, and
enable playback before the greeting. Do not give the browser the Microsoft AI
key, server API secret or microphone access. Run only this unnamed demo agent
against the local server so that automatic dispatch chooses it. This example
uses the room transport; use `dev`, not the local-device `console` mode.

The same example can run from a clean environment with the built wheel and
the declared minimum released SDK, rather than relying on editable sources:

```sh
uv build --package livekit-plugins-microsoft-ai --out-dir /tmp/microsoft-ai-dist
uv venv /tmp/microsoft-ai-room
uv pip install --python /tmp/microsoft-ai-room/bin/python \
  /tmp/microsoft-ai-dist/livekit_plugins_microsoft_ai-1.8.3-py3-none-any.whl \
  'livekit-agents==1.8.3'
/tmp/microsoft-ai-room/bin/python \
  examples/microsoft/microsoft_ai_tts_room.py dev --no-reload
```

### Full STT/LLM/TTS agent

See `examples/microsoft/microsoft_ai_agent.py` for an AgentSession using the
existing OpenAI LLM, bundled VAD and this STT/TTS package. Its OpenAI credential
is used by the LLM only. This full agent requires STT access as well as TTS.
It cannot run with only an Azure Speech TTS key: `microsoft_ai.STT` also needs
its separate endpoint/model/credentials, and the LLM needs `OPENAI_API_KEY`.
The CLI's `console` mode removes the Cloud requirement, not those provider
requirements. Use the TTS-only room example when STT/LLM access is unavailable.

```sh
uv run --package livekit-agents --extra microsoft-ai --extra openai --no-default-groups \
  python examples/microsoft/microsoft_ai_agent.py console
```

To use an external config file with the agent, set `MICROSOFT_AI_ENV_FILE` to
its path first. Keep the LLM's `OPENAI_API_KEY` separate; the Microsoft AI config
loader does not copy unrelated variables into the process environment.

### Direct endpoint smoke test

`examples/microsoft/microsoft_ai_smoke.py` is a direct, explicit-opt-in smoke path
without LiveKit Cloud or an LLM. It sends at most one short TTS request and one
user-approved speech fixture, with no automatic retries, recording, transcript
printing, audio playback or load testing. TTS uses the Azure Speech subscription
key; STT uses the explicitly configured auth selector (Bearer by default, or
raw `api-key`). The smoke script reads it from the same external dotenv file;
no key or header value belongs in CLI arguments. The whole smoke run is bounded
to 50 seconds.

```sh
uv run --package livekit-plugins-microsoft-ai --no-default-groups \
  python examples/microsoft/microsoft_ai_smoke.py --help
```

After confirming the exact TTS endpoint and approving the fixed short text,
opt in to **TTS only**, with no STT URL, key, model or fixture required:

```sh
uv run --package livekit-plugins-microsoft-ai --no-default-groups \
  python examples/microsoft/microsoft_ai_smoke.py --run-live --tts \
  --env-file /path/outside/checkout/endpoints.env
```

Only when STT access and its protocol are separately confirmed, test a tiny
approved STT fixture (add `--tts` to test both services):

```sh
uv run --package livekit-plugins-microsoft-ai --no-default-groups \
  python examples/microsoft/microsoft_ai_smoke.py --run-live \
  --env-file /path/outside/checkout/endpoints.env \
  --stt-wav /path/outside/checkout/approved-speech.wav \
  --expected-text-file /path/outside/checkout/approved-expected.txt
```

The WAV must be PCM16 mono at 16 kHz, nonempty and no longer than five seconds.
The script checks the complete expected words, ignoring only case/punctuation.
It sends no additional silence and does not fabricate a final transcript from
an intermediate hypothesis. Repeat manually with a separately approved short,
non-chunk-aligned clip to check the deployment's tail handling; one successful
clip is not a universal tail guarantee. `--tts` sends only the fixed text
`Hello, this is a Microsoft AI voice test.` It validates framing, not subjective
voice quality. Omit either service's flags to test only the other.
The reported elapsed time covers the client synthesis call through complete
stream closure; it is **not model TTFA** because audio is emitted only after
the complete WAV response has been validated.

Focused hermetic tests:

```sh
uv sync --package livekit-plugins-microsoft-ai --no-default-groups
uv sync --only-group dev --no-default-groups --inexact
uv run --no-sync pytest tests/test_microsoft_ai_stt.py tests/test_microsoft_ai_tts.py --unit
```

These tests use fake sockets/HTTP responses and synthetic audio only. They
validate client behavior, not real endpoint compatibility.
Validate each service independently for a new deployment: confirm URLs,
credentials and auth transport, model/voice IDs, and event/request/response
schemas. Successful Azure Speech TTS validation does not validate STT or its
backend audio-tail finalization contract.
