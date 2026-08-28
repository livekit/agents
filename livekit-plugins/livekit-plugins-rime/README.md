# Rime plugin for LiveKit Agents

Support for voice synthesis with the [Rime](https://rime.ai/) API.

See [https://docs.livekit.io/agents/integrations/tts/rime/](https://docs.livekit.io/agents/integrations/tts/rime/) for more information.

## Installation

```bash
pip install livekit-plugins-rime
```

## Pre-requisites

You'll need an API key from Rime. It can be set as an environment variable: `RIME_API_KEY`

## Streaming Coda WebSocket API

The Rime v1 WebSocket protocol accepts streaming text and returns audio before the input turn
is complete. The plugin aggregates input fragments into complete sentences before it sends them
to Coda. All sentences in one LiveKit output turn use one continuous Coda context.

```python
import os

from livekit.plugins import rime

tts = rime.TTS(
    websocket_url="wss://api.rimetts.com/coda/v1/coda/ws",
    speaker="astra",
    api_key=os.environ["RIME_API_KEY"],
)
```

Pass the active Coda WebSocket endpoint explicitly. The presence of `websocket_url` selects Coda,
WebSocket streaming, and the v1 JSON protocol. The speaker defaults to `astra`. The plugin uses
`livekit.agents.tokenize.blingfire.SentenceTokenizer` by default and configures it to emit one
complete sentence at a time. Pass `tokenizer` to select another LiveKit sentence tokenizer. A
custom tokenizer must emit complete sentence units that are safe for Coda text normalization.

One LiveKit stream uses one continuous Rime synthesis context. These stream methods map to the
Rime lifecycle as follows:

| LiveKit method | Rime operation | Result |
| --- | --- | --- |
| `stream.flush()` | `flush` | Speak pending text and keep the context open. |
| `stream.end_input()` | `end` | Finalize input and wait for `done`. |
| `stream.aclose()` | `cancel` | Cancel synthesis if the context is still active. |

You can send more text after `flush()`. A flush does not cause a `done` event and does not start a
new synthesis context. A flush drains the current sentence tokenizer before it asks Coda to speak
pending text. Later text continues in the same Coda context with a new tokenizer stream. The
tokenizer normally sends complete sentences. Calling `flush()` commits any buffered fragment,
including an incomplete sentence. Call `flush()` only after a complete sentence or a stable clause.

The first v1 implementation has these limits:

- WebSocket v1 supports the `coda` model only.
- It requests raw `audio/pcm` data.
- It does not provide aligned word timestamps.
- `speed_alpha` is not supported. Use `time_scale_factor` to control speed.
- The adapter uses the JSON `rime.v1.json` WebSocket subprotocol.
- The adapter reuses WebSocket connections between sequential LiveKit streams. It does not run
  concurrent Coda contexts on one WebSocket. Concurrent LiveKit streams use separate pooled
  WebSocket connections.
