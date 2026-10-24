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
`POST https://api.giggy.ai/v1/audio/speech` and delivers progressive, signed PCM16
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

`giggy.STT` accepts live audio through
`wss://api.giggy.ai/v1/audio/transcriptions`. It reports `streaming=True` and
`interim_results=True`. An API key is required; no voice UUID is needed.
This native streaming implementation is an unreleased source change and requires
Giggy's matching WebSocket preparation handler; it is not in a published package.

```python
from livekit.agents import AgentSession
from livekit.plugins import giggy

session = AgentSession(
    stt=giggy.STT(language="en"),
    tts=giggy.TTS(voice="YOUR_GIGGY_VOICE_UUID"),
)
```

Supply your existing LLM configuration. LiveKit feeds frames directly to the
provider's native stream. The included Silero endpoint detector defaults to
100 ms silence, 200 ms sustained speech, onset threshold 0.7 and continuation
threshold 0.35. Pass `vad=your_detector` to `giggy.STT` to select a compatible
LiveKit VAD. Keep application turn-detection settings fixed during comparisons.

The stream resamples mono frames to 16 kHz PCM16, retains up to one second of
real pre-roll and sends real audio while speech is happening. During candidate
silence it sends the existing canonical `decode_end_sample` through a `prepare`
control, allowing decoding to overlap endpoint confirmation. Resumed speech
withdraws that hint. All captured audio, including the real tail, is still sent;
the hint neither ends input nor changes accepted-duration billing.

Each utterance requires 0.5–30 seconds of real PCM; shorter captures wait for
real audio or fail at clean EOF, and longer turns fail explicitly. No padding,
transport fallback, automatic replay or automatic retry is used. Up to eight
operations can finish concurrently with at most 30 seconds of aggregate queued
PCM. Final transcripts retain utterance order. Native interim/final events are
validated against generation, attempt, sequence and the final input ACK. Final
text is delivered before saved-result verification; keep consuming the stream
and surface verification errors even after final text was delivered.

`stream.end_input()` cleanly drains captured input and verifies saved results.
`stream.aclose()` aborts unfinished operations. `await recognizer.aclose()` closes
all streams and explicit recording requests, rejects new work and leaves borrowed
HTTP sessions open. The overall per-turn deadline defaults to 90 seconds;
`APIConnectOptions.timeout` bounds connection acquisition and idle socket reads.

Language defaults to detection. A supported name or BCP-47 code, such as
`"English"` or `"en-US"`, can be supplied at construction or per stream. No word
alignment or model confidence is fabricated.

For an already completed recording, explicit
`await recognizer.recognize(audio_frame)` retains the WAV/SSE interface and
returns after canonical completion and saved text/duration verification. It
accepts 0.5 seconds–12 hours and stages bounded WAV chunks in a temporary file.
This recording latency is distinct from live speech-end-to-final latency.
It is not a fallback for a failed live stream.

Use credential-free borrowed sessions without default auth headers or cookies.
Signed artifacts are fetched without the Giggy key and restricted to canonical
HTTPS Cloudflare R2 account hosts on port 443; artifact redirects and URL
credentials are rejected. Input ACK, canonical completion, saved text and accepted
duration are verified. Offline tests do not establish production latency or
live-room behavior.

STT uses Streaming pricing, currently 8,000 credits per input audio hour,
rounded up per request. Paid production recognition and live-room testing are
separate from the offline tests below.

Offline tests do not require Giggy credentials or call the production service:

```sh
uv run --package livekit-plugins-giggy pytest -q livekit-plugins/livekit-plugins-giggy/tests/
```

This plugin contribution is separate from LiveKit Inference onboarding and
does not establish availability in Agent Builder.
