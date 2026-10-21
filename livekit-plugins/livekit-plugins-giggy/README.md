# Giggy speech plugin for LiveKit Agents

Native Python integration with the [Giggy speech API](https://giggy.ai/docs/speech-api).
This contribution is not an officially released package until upstream accepts and releases it.

For development, install this directory from the Agents workspace:

```sh
uv pip install -e livekit-plugins/livekit-plugins-giggy
```

Set `GIGGY_API_KEY` and `GIGGY_VOICE_ID` to your API key and voice UUID, or pass
them explicitly:

```python
from livekit.plugins import giggy

provider = giggy.TTS(voice="YOUR_GIGGY_VOICE_UUID", speed=1.0)
```

Use `provider` as the TTS component of your existing `AgentSession`.
The model is `giggyspeech`; synthesis submits complete text to
`POST https://giggy.ai/v1/audio/speech` and delivers progressive, signed PCM16
little-endian audio at 24 kHz, mono. Speed ranges from 0.25 to 4.

Incremental text input and aligned transcripts are not supported. LiveKit's
capabilities therefore report `streaming=False`, even though audio bytes arrive
progressively over HTTP. No WebSocket or alternate provider is used.

Synthesis uses paid Streaming admission. Automatic retries are disabled even
when the caller requests them, because an interrupted request may already have
incurred a charge. No `Idempotency-Key` is sent. Cancellation closes the HTTP
response. An injected `aiohttp.ClientSession` remains owned by the caller;
otherwise the plugin uses LiveKit's shared HTTP session.

`await provider.aclose()` closes all of the provider's synthesis streams and
rejects new synthesis, while leaving caller-owned and shared HTTP sessions open.
`APIConnectOptions.timeout` controls connection acquisition, socket connection,
and the idle timeout between audio reads; it is not a total utterance deadline.
Choose a value that allows for the service's first-audio wait.

## Speech recognition

`giggy.STT` recognizes complete utterances through the canonical V21
`POST /v1/audio/transcriptions` WAV endpoint in paid Streaming mode.
It requires `GIGGY_API_KEY`; no voice UUID is needed for recognition.

```python
from livekit.agents import AgentSession
from livekit.plugins import giggy, silero

session = AgentSession(
    stt=giggy.STT(language="en"),
    vad=silero.VAD.load(),
    tts=giggy.TTS(voice="YOUR_GIGGY_VOICE_UUID"),
)
```

Supply your existing LLM configuration to the session. Install
`livekit-plugins-silero` separately when using this VAD example. AgentSession's
standard VAD stream adapter collects each utterance and calls the recognizer.
Direct callers can also use `await recognizer.recognize(audio_frame)`.

Recognition accepts mono LiveKit audio frames, resamples to native 16 kHz PCM16
when necessary, and uploads a complete WAV without padding. Audio must be at
least 0.5 seconds and at most 12 hours, matching V21's uploaded-recording limit.
Encoding stages the WAV in a temporary file using bounded audio chunks, and
aiohttp streams the file upload without merging the full recording in memory.
The separate live-microphone protocol's 30-second limit does not apply here.
Configure VAD to capture valid utterances. The default overall recognition
deadline is 90 seconds; `recognition_timeout` can be set explicitly.
`APIConnectOptions.timeout` controls connection acquisition, socket connection,
and idle reads. Use a deadline appropriate for the recording length and service wait.

Language defaults to automatic detection. Set a supported language name or BCP-47
code, such as `"English"` or `"en-US"`, in the constructor or per recognition call.
Detected languages are returned in the final transcript's `source_languages`.

The provider reports `streaming=False` and `interim_results=False`: its input is
a completed utterance. It consumes cumulative SSE snapshots internally and
returns a final transcript only after canonical completion and verification of
the saved transcript text and decoded audio duration. It does not expose Giggy's
separate live-microphone WebSocket input protocol or word alignment.

Each recognition is admitted once with its own `client_request_id`; automatic
retries and redirects are disabled. Closing or cancelling recognition closes
the active SSE response, requesting cancellation through the existing server
rail. Completion or GPU cancellation timing is not benchmarked by offline tests.
`await recognizer.aclose()` cancels active recognition and rejects new requests.
Borrowed sessions remain open. Use a session without default authentication,
credential headers, or cookies: signed transcript artifacts are fetched without
the API key. Artifact destinations are restricted to HTTPS on port 443 using
Cloudflare R2's canonical `<32-hex-account-id>.r2.cloudflarestorage.com` hosts,
as used by Giggy's result signer. Redirects and URL credentials are rejected.
The SDK exposes no alternate recognition provider or transport.

STT uses Streaming pricing, currently 8,000 credits per input audio hour,
rounded up per request. Paid production recognition and live-room testing are
separate from the offline tests below.

Offline tests do not require Giggy credentials or call the production service:

```sh
uv run --package livekit-plugins-giggy pytest -q livekit-plugins/livekit-plugins-giggy/tests/
```

This plugin contribution is separate from LiveKit Inference onboarding and
does not establish availability in Agent Builder.
