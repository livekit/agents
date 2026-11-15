# Rime plugin for LiveKit Agents

Support for voice synthesis with the [Rime](https://rime.ai/) API.

See [https://docs.livekit.io/agents/integrations/tts/rime/](https://docs.livekit.io/agents/integrations/tts/rime/) for more information.

## Installation

```bash
pip install livekit-plugins-rime
```

## Pre-requisites

You'll need an API key from Rime. It can be set as an environment variable: `RIME_API_KEY`

## Streaming and one-shot synthesis

`stream()` speaks over Rime's WebSocket v1 API; `synthesize()` uses the HTTP API. The endpoint
follows the model: Coda streams from `wss://coda.api.rime.ai/ws` (the default) and Mist v3 from
`wss://mist.api.rime.ai/ws`. Mist v2 synthesizes over HTTP only.

```python
import os

from livekit.plugins import rime

tts = rime.TTS(model="coda", speaker="lyra", api_key=os.environ["RIME_API_KEY"])
```

A dedicated endpoint is passed as `websocket_url`, with its `model` when its path is only `/ws`:

```python
tts = rime.TTS(
    websocket_url="wss://tigerstripe-dialpad.aws-us-east-1.whiteglove.rime.ai/ws",
    model="coda",
    api_key=os.environ["RIME_API_KEY"],
)
```

The plugin sends the Rime API key to the endpoint. It trusts `rime.ai` and its subdomains by
default. To use an endpoint on a different domain, confirm that you trust it and set
`allow_custom_endpoint=True`. This opt-in also applies to `base_url`.

`websocket_protocol` is `binary` (the `rime.v1.binary` subprotocol, protobuf frames, the
default) or `json` (`rime.v1.json`, proto3 JSON text frames).

The speaker defaults to `lyra` for Coda and `cove` for Mist. The plugin sends the resolved
sample rate in every request so it matches the audio emitter's rate; pass `sample_rate` to
override the model default.

`audio_format` selects the streamed audio's format: `audio/pcm` (the default), `audio/wav`,
`audio/mpeg`, `audio/ogg;codecs=opus`, `audio/webm;codecs=opus` or `audio/pcmu`. HTTP synthesis
returns PCM.

`time_scale_factor` controls the speed. The Coda generation controls (`repetition_penalty`,
`temperature`, `top_p`, `max_tokens`) apply to HTTP synthesis only.

Each segment of a stream is one Rime context: `start`, its text, then `end`, answered by
`done`. Closing a stream mid-segment sends `cancel`. Sockets are reused by later streams, one
stream at a time each. The v1 protocol sends no word timestamps.
