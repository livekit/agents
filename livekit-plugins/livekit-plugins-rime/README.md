# Rime plugin for LiveKit Agents

Support for voice synthesis and speech recognition with [Rime](https://rime.ai/).

See [https://docs.livekit.io/agents/integrations/tts/rime/](https://docs.livekit.io/agents/integrations/tts/rime/) for more information.

## Installation

Requires Python 3.11 or newer for both TTS and STT. Applications using Python 3.10
must upgrade Python or keep a compatible older plugin release.

```bash
pip install livekit-plugins-rime
```

## Pre-requisites

You'll need an API key from Rime. It can be set as an environment variable: `RIME_API_KEY`

## Speech recognition

STT uses the Rime Python SDK. Install the `stt` extra:

```bash
pip install 'livekit-plugins-rime[stt]'
```

The extra installs `rimelabs-sdk>=0.1.0a18`, which provides unary transcription,
revised-hypothesis streaming, and append-fragment streaming.

```python
from livekit.agents import AgentSession
from livekit.plugins import rime, silero

session = AgentSession(
    vad=silero.VAD.load(),
    stt=rime.STT(
        language="en",
        mode="written",  # or "verbatim" for spoken forms
        context_terms=["LiveKit", "Rime"],
    ),
    # Configure your LLM and TTS as usual.
)
```

Install `livekit-plugins-silero` to use the VAD in this example. LiveKit automatically wraps
this non-streaming recognizer with `stt.StreamAdapter`: VAD detects each completed speech
segment, then the adapter submits that segment for transcription. Results arrive after the
segment ends. To receive interim transcripts during speech, enable native streaming below.

For an existing recording, call `await recognizer.recognize(buffer=audio_frame)` with a
`rtc.AudioFrame` or a list of frames in the same format. The plugin mixes channels to mono,
resamples to 16 kHz, and sends PCM16 WAV bytes to the SDK. Rime accepts at most 16 MiB of WAV
audio per request. The result is a `FINAL_TRANSCRIPT` speech event containing the text, resolved
language, and request ID. Word timestamps, speaker labels, and confidence scores are not supplied.

`language` is explicit (English by default); it is not automatic language detection. Use
`update_options(language=..., mode=..., context_terms=...)` to change subsequent requests, or
`recognize(..., language=...)` to override the language for a single request. Session keyterms
are merged with `context_terms`. LiveKit's STT connection options control the request timeout
and retries; only SDK timeouts and service-unavailable errors are retried. Close standalone
recognizers with `await recognizer.aclose()` or use `async with rime.STT(...) as recognizer`.

## Native streaming speech recognition

Pass a VAD directly to `rime.STT` to upload audio while the user speaks. One SDK
stream handles each VAD utterance. Select `revised_hypotheses` (the default) for
replacement partials or `append_fragments` for permanent fragments:

```python
from livekit.agents import APIConnectOptions, AgentSession
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import rime, silero

detector = silero.VAD.load()
session = AgentSession(
    vad=detector,
    stt=rime.STT(vad=detector, output_contract="append_fragments"),
    conn_options=SessionConnectOptions(
        stt_conn_options=APIConnectOptions(timeout=120, max_retry=0),
    ),
    # Configure your LLM and TTS as usual.
)
```

Append-fragment support is available in the published `rimelabs-sdk>=0.1.0a18`
installed by the `stt` extra; no SDK source checkout is required.

The plugin concatenates deltas exactly, preserving spaces and newlines, and sends
cumulative `INTERIM_TRANSCRIPT` snapshots to LiveKit. Revised hypotheses replace
the previous snapshot. Only SDK-confirmed completion emits `FINAL_TRANSCRIPT`;
individual fragments do not end the user turn. Append mode currently uses
five-second windows, so short utterances may have no update before speech ends.
Revised mode generally provides earlier provisional text for voice agents.

Use mono input; LiveKit resamples streaming audio to 16 kHz. The VAD must provide
initial speech frames in `START_OF_SPEECH` and subsequent new frames in
`INFERENCE_DONE` (as Silero does). The complete `END_OF_SPEECH` buffer is not sent
again. `flush()` drains the segment and resets the detector; `end_input()` also
finishes an active utterance. `aclose()` cancels it without creating a final.

Options and merged session keyterms are captured at each utterance start.
Streaming timeouts include speaking and upload time: use a sufficiently long
`stt_conn_options.timeout` (the example uses 120 seconds). Live audio is never
replayed automatically, regardless of `max_retry`. SDK errors retain the request
ID without including provider messages. Later utterances upload while earlier
requests finish. Speech boundaries and transcripts are published together in
utterance order: start, interim updates, final transcript, then end. The end event
waits for successful transcription; its timestamp still reflects when speech ended.
While waiting to publish, only the latest cumulative interim snapshot is retained
per utterance. Supply a VAD to `AgentSession` as shown above for immediate speech
activity and interruption detection while STT events await earlier results.
Pending upload audio is bounded to ten seconds across all requests, and at most eight utterances may await completion/publication;
overflow fails explicitly. SDK/service duration and size limits still
apply. The caller retains ownership of the supplied VAD.

## Streaming WebSocket v1 API

The Rime v1 WebSocket protocol accepts streaming text and returns audio before the input turn
is complete. The plugin aggregates input fragments into complete sentences before it sends them
to Rime. All sentences in one LiveKit output turn use one continuous synthesis context.

```python
import os

from livekit.plugins import rime

tts = rime.TTS(
    websocket_url="wss://coda.api.rime.ai/ws",
    model="coda",
    speaker="lyra",
    api_key=os.environ["RIME_API_KEY"],
)
```

Pass the active model WebSocket endpoint explicitly. The presence of `websocket_url` selects
WebSocket v1 streaming. Public model endpoints use port 443 and the `/ws` path.
Pass `model="coda"` for `wss://coda.api.rime.ai/ws`, which serves Coda v2.
Pass `model="mistv3"` for `wss://mist.api.rime.ai/ws`.
`model` is required when constructing any WebSocket v1 client. The plugin does not infer
the model from the URL hostname or path.

Dedicated deployment endpoints also require `model`:

```python
tts = rime.TTS(
    websocket_url="wss://voice.customer.example/ws",
    model="coda",
    allow_custom_endpoint=True,
    api_key=os.environ["RIME_API_KEY"],
)
```

The plugin sends the Rime API key to the endpoint. It trusts `rime.ai` and its subdomains by
default. To use an endpoint on a different domain, confirm that you trust it and set
`allow_custom_endpoint=True`. This opt-in also applies to `base_url`.

The `websocket_protocol` option accepts `binary` or `json`. It defaults to `binary`, which uses the
`rime.v1.binary` subprotocol and protobuf binary frames. Set `websocket_protocol="json"` to use the
`rime.v1.json` subprotocol and canonical proto3 JSON text frames.

The speaker defaults to `lyra` for Coda and `cove` for Mist across all transports.
For HTTP and WS3, an omitted model defaults to Coda. An explicit speaker is preserved.
The plugin uses
`livekit.agents.tokenize.blingfire.SentenceTokenizer` by default and configures it to emit one
complete sentence at a time. Pass `tokenizer` to select another LiveKit sentence tokenizer. A
custom tokenizer must emit complete sentence units that are safe for Rime text normalization.

The plugin sends the resolved sample rate in every HTTP, WS3, and WebSocket v1 request so it
matches the audio emitter's rate. Pass `sample_rate` to override the model default.

If your configuration relied on a model in the URL path, add `model` to the constructor.
Custom URL paths pass through without model detection.

Select the Rime output format with `audio_format`. It defaults to `audio/pcm`. The adapter supports
all canonical formats from the Rime interface:

- `audio/pcm`
- `audio/wav`
- `audio/mpeg`
- `audio/ogg;codecs=opus`
- `audio/webm;codecs=opus`
- `audio/pcmu`

Both WebSocket protocols support every format. The binary protocol sends the encoded audio bytes
directly. The JSON protocol sends the same bytes as base64 text.

PCMU uses the selected sample rate. Choosing `audio/pcmu` keeps the model default, which is
24,000 Hz for Coda. The plugin decodes PCMU to 16-bit PCM at the same rate. To request 8,000 Hz,
set `sample_rate` explicitly:

```python
tts = rime.TTS(
    websocket_url="wss://coda.api.rime.ai/ws",
    model="coda",
    audio_format="audio/pcmu",
    sample_rate=8000,
    api_key=os.environ["RIME_API_KEY"],
)
```

The v1 interface uses `time_scale_factor` for speed control. The deprecated `speed_alpha` option
applies only to the older Rime interfaces.

One LiveKit stream uses one continuous Rime synthesis context. These stream methods map to the
Rime lifecycle as follows:

| LiveKit method | Local action | Rime operation |
| --- | --- | --- |
| `stream.push_text()` | Buffer and sentence-tokenize text. | Send `text` for each completed sentence. |
| `stream.flush()` | Release the current tokenizer buffer and keep the context open. | Send released content as `text`; no `flush` operation exists. |
| `stream.end_input()` | Drain final text and finalize input. | Send `end`. |
| `stream.aclose()` | Stop active synthesis. | Send `cancel` when needed. |

You can send more text after `flush()`. The same Rime context remains active, and an input pause
needs no wire message. Only `end_input()` ends normal input and causes Rime to send `done`.
Each outgoing `text` value must be a stable sentence-sized unit. Calling `flush()` drains any
buffered fragment, so call it only after a complete sentence or a stable clause.

The v1 implementation has these limits:

- It does not provide aligned word timestamps.
- The adapter reuses WebSocket connections between sequential LiveKit streams. It does not run
  concurrent contexts on one WebSocket. Concurrent LiveKit streams use separate pooled
  WebSocket connections.
