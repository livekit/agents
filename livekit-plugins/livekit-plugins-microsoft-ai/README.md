# Microsoft AI plugin for LiveKit Agents

## Service availability

**TTS** uses Microsoft's public, documented
[MAI-Voice](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-voices)
path: SSML posted to the Azure Speech REST synthesis endpoint with a Speech
resource key. It has been verified with `en-US-Harper:MAI-Voice-2-Flash` and
`en-US-Harper:MAI-Voice-2.1-Flash` (PCM16 mono at 24 kHz) on an `eastus` Speech
resource, including playback through a local LiveKit room. MAI voices are in
public preview.

**STT** has no publicly documented Microsoft endpoint. The plugin speaks a
realtime streaming transcription websocket protocol (described below), but
Microsoft documents
[MAI-Transcribe](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe)
only through:

- the Speech **fast transcription** REST API
  (`speechtotext/transcriptions:transcribe` with `enhancedMode.model`), which
  is file-based, not streaming, and is **not** used by this plugin; and
- **Voice Live**, as input-audio transcription inside a Voice Live session.

Neither is a streaming endpoint that `microsoft_ai.STT` can connect to. Use
`microsoft_ai.STT` only with a MAI realtime transcription endpoint that
Microsoft has provided to you directly, and confirm its contract against the
protocol below. **Do not point it at an Azure OpenAI endpoint**
(`*.openai.azure.com` or `/openai/v1/realtime`): MAI-Transcribe is not
deployable there, so transcription fails with `DeploymentNotFound`.

Supported MAI-Transcribe models are `MAI-Transcribe-2` and `MAI-Transcribe-1.5`;
`MAI-Transcribe-1` was deprecated on August 20, 2026.

**Decisions** use
[Microsoft-Decision-1](https://ai.azure.com/catalog/models/Microsoft-Decision-1)
deployed in Microsoft Foundry, through the resource's System One endpoint
(`POST <resource endpoint>/providers/microsoft/v1/systemone`). See
[Decision model contract](#decision-model-contract).

None of this establishes access in every resource or region, recognition
accuracy across inputs/languages, every backend tail boundary, long-session
reliability, physical microphone behavior, or subjective voice quality.
Hermetic tests cover the client lifecycle and VAD ordering. None of these
results is a model-latency benchmark. An Azure Speech TTS resource/key does
**not** establish access to any STT endpoint.

The package requires `livekit-agents>=1.8.3`.

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
| `MICROSOFT_AI_TTS_MODEL` | `TTS(model=...)` (optional with a full voice ID) |
| `MICROSOFT_AI_TTS_VOICE` | `TTS(voice=...)` |
| `MICROSOFT_AI_TTS_SAMPLE_RATE` | `TTS(sample_rate=...)` |
| `MICROSOFT_AI_DECISION_URL` | `DecisionModel(url=...)` |
| `MICROSOFT_AI_DECISION_API_KEY` | `DecisionModel(api_key=...)` |
| `MICROSOFT_AI_DECISION_AUTH_HEADER` | `DecisionModel(auth_header=...)`: `api-key` or `Authorization` |
| `MICROSOFT_AI_DECISION_MODEL` | `DecisionModel(model=...)` (Foundry deployment name) |
| `MICROSOFT_AI_ENV_FILE` | `STT(env_file=...)`, `TTS(env_file=...)`, `DecisionModel(env_file=...)` |

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
For an STT endpoint that expects a raw resource key, explicitly set
`MICROSOFT_AI_STT_AUTH_HEADER=api-key` (or `auth_header="api-key"`); it sends the
raw credential from `MICROSOFT_AI_STT_API_KEY` as the `api-key` header, with no
Bearer prefix. The selector accepts only the exact values `Authorization` and
`api-key`; an empty/unknown value is an error. No auth fallback or automatic
scheme detection occurs, and credentials are never added to the URL.

Explicit `headers` (including `{}`) override the STT selector and credential
environment settings and cannot be combined with `api_key` or `auth_header`
constructor arguments. Use them only for a confirmed alternate authentication
scheme. No credentials are read from OpenAI/Azure variables. Caller-supplied
`http_session` objects are borrowed; otherwise STT and TTS own lazy sessions and
close them in `aclose()`, and `DecisionModel` uses the job's HTTP session (outside
a job, wrap calls in `async with utils.http_context.open()`).

Keep actual connection information in your environment or a user-selected
local dotenv file **outside the checkout** (or a deliberately ignored file).
The providers load a file only when `env_file` or `MICROSOFT_AI_ENV_FILE` selects
one; there is no automatic `.env` discovery. Files are parsed with python-dotenv,
without shell sourcing, variable interpolation, environment mutation or value
logging. Precedence is constructor argument, then process environment, then the
selected file. An empty required value fails instead of falling back silently.
Use owner-only file permissions and never copy this file into commits.

## STT contract and lifecycle

See [Service availability](#service-availability) first: there is no publicly
documented MAI streaming STT endpoint, and Azure OpenAI endpoints do not serve
MAI-Transcribe. Configure the full URL of the endpoint Microsoft provided in
`MICROSOFT_AI_STT_URL` and its model identifier in `MICROSOFT_AI_STT_MODEL`.
The model is sent in `session.audio.input.transcription.model`, not added as a
URL query parameter. The plugin sends the URL unchanged; it does not add a
`model=`, `deployment` or `api-version` query. Do not put a key in the URL.
Validate the endpoint's audio-rate and transcript-event compatibility against
the contract below before relying on it.

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
   followed by end-of-speech. The completion may correct or retract earlier
   hypotheses; an empty completion finishes the item without inventing user
   text. The socket stays open for subsequent utterances.

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
There is no invented padding rule. The acknowledged completion is authoritative,
even when it removes an interim hypothesis. Comparing final and revisable text
cannot distinguish legitimate revision from recognition or audio-tail loss;
discarded hypotheses are never appended to the final. Verify the full expected
transcript, particularly the last word, for both a short clip and a
non-chunk-aligned tail. Obtain a documented backend drain/flush mechanism if
commit does not decode the tail.

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
following the standard public-cloud Azure Speech convention. The plugin does
not check region availability: MAI voices are offered only in some Speech
regions (see the **MAI voices** column of
[Speech service regions](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/regions?tabs=tts)),
and the resource key must belong to the selected region. Sovereign clouds and
custom/private deployments require an explicit full URL. There is no automatic
region detection, failover or redirection to a different region.

The SSML voice ID is `name:model`, such as `en-US-Harper:MAI-Voice-2-Flash`.
Set `voice` to the name (`en-US-Harper`) and `model` to the model
(`MAI-Voice-2-Flash`), and the plugin joins them. A full ID in `voice` is also
accepted; `model` is then optional, and when set it is checked against the ID's
suffix, case-insensitively. Availability still depends on the resource and
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

## Decision model contract

`microsoft_ai.DecisionModel` answers a batch of probability, choice, and score
decisions about the same input in one request:

```python
from livekit.agents import llm
from livekit.plugins import microsoft_ai

model = microsoft_ai.DecisionModel()
response = await model.evaluate(
    input=chat_ctx,
    decisions={
        "handoff": llm.Probability("The caller requests a human."),
        "intent": llm.Choice("Intent?", options={"booking": "Reservation", "other": "Other"}),
    },
)
```

Configure the full System One URL of your Foundry resource in
`MICROSOFT_AI_DECISION_URL`, for example
`https://<resource>.services.ai.azure.com/providers/microsoft/v1/systemone`, and
the deployment name in `MICROSOFT_AI_DECISION_MODEL`. The resource key is sent
as the raw `api-key` header by default; set `auth_header="Authorization"` to send
an Entra ID token as `Authorization: Bearer ...`.

The request body is `{"model", "state", "questions"}`: a conversation is sent as
its transcript in `state`, and each decision as a named `noul`, `choice` or
`score` question. An answer the model declines or that fails validation is
reported in `DecisionResponse.errors` without dropping the rest of the batch.
Non-2xx responses raise `APIStatusError` with the response body and request ID;
408, 429 and 5xx are retried according to `conn_options`.

## Validation


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
