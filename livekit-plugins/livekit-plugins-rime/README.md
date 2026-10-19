# Rime plugin for LiveKit Agents

Support for voice synthesis with the [Rime](https://rime.ai/) API.

See [https://docs.livekit.io/agents/integrations/tts/rime/](https://docs.livekit.io/agents/integrations/tts/rime/) for more information.

## Installation

```bash
pip install livekit-plugins-rime
```

## Pre-requisites

You'll need an API key from Rime. It can be set as an environment variable: `RIME_API_KEY`

## Streaming WebSocket v1 API

The Rime v1 WebSocket protocol accepts streaming text and returns audio before the input turn
is complete. The plugin aggregates input fragments into complete sentences before it sends them
to Rime. All sentences in one LiveKit output turn use one continuous synthesis context.

WebSocket v1 requires both `websocket_url` and `model` at construction. The URL selects
the deployment. The model sets the plugin's defaults and available options.

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

Use `model="coda"` with `wss://coda.api.rime.ai/ws`, or `model="mistv3"` with
`wss://mist.api.rime.ai/ws`. Both public endpoints use port 443.

If your configuration relied on a model in the URL path, add `model` to the constructor.
The plugin does not read the model from the URL hostname or path.

Custom deployments can use any URL path. Replace this example URL with your deployment URL:

```python
tts = rime.TTS(
    websocket_url="wss://voice.customer.example/custom/stream",
    model="coda",
    allow_custom_endpoint=True,
    api_key=os.environ["RIME_API_KEY"],
)
```

The plugin sends the Rime API key to the endpoint. It trusts `rime.ai` and its subdomains by
default. To use an endpoint on a different domain, confirm that you trust it and set
`allow_custom_endpoint=True`. This opt-in also applies to `base_url`.

`websocket_protocol` accepts `binary` or `json`. It defaults to `binary`, which uses the
`rime.v1.binary` subprotocol and protobuf binary frames. Set `websocket_protocol="json"` to use the
`rime.v1.json` subprotocol and canonical proto3 JSON text frames.

For Coda, the default speaker is `lyra`. Mist uses `cove`. Pass `speaker` to use another voice.

By default, the plugin uses `livekit.agents.tokenize.blingfire.SentenceTokenizer` and configures
it to emit one complete sentence at a time. Pass `tokenizer` to select another LiveKit sentence
tokenizer. A custom tokenizer must emit complete sentence units that are safe for Rime text
normalization.

The plugin sends the resolved sample rate in every HTTP, WS3, and WebSocket v1 request so it
matches the audio emitter's rate. Pass `sample_rate` to override the model default.

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
