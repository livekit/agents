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

```python
import os

from livekit.plugins import rime

tts = rime.TTS(
    websocket_url="wss://api.rime.ai/coda/ws",
    speaker="lyra",
    api_key=os.environ["RIME_API_KEY"],
)
```

Pass the active model WebSocket endpoint explicitly. The presence of `websocket_url` selects
WebSocket v1 streaming. For a route such as `/coda/ws`, the plugin reads the model from the path
segment before `/ws`. Do not also pass `model` for this type of route.

Some dedicated Rime endpoints end with `/ws` and do not contain a model path segment. Pass
`model` for these endpoints:

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

The `websocket_protocol` option accepts `binary` or `json`. It defaults to `binary`, which uses the
`rime.v1.binary` subprotocol and protobuf binary frames. Set `websocket_protocol="json"` to use the
`rime.v1.json` subprotocol and canonical proto3 JSON text frames.

The speaker defaults to `lyra` for Coda and `cove` for Mist across all transports. This also
applies when the model is omitted and defaults to Coda. An explicit speaker is preserved.
The plugin uses
`livekit.agents.tokenize.blingfire.SentenceTokenizer` by default and configures it to emit one
complete sentence at a time. Pass `tokenizer` to select another LiveKit sentence tokenizer. A
custom tokenizer must emit complete sentence units that are safe for Rime text normalization.

The plugin sends the resolved sample rate in every HTTP, WS3, and WebSocket v1 request so it
matches the audio emitter's rate. Pass `sample_rate` to override the model default.

The public WebSocket v1 endpoint currently supports Coda. The future Mist route will use
`/mist/ws`, not `/mistv3/ws`.

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
    websocket_url="wss://api.rime.ai/coda/ws",
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

## Prism realtime

Prism combines speech input, reasoning, tool calls, and speech output in one
realtime model. The plugin calls the Python Rime SDK for all Prism protocol work.
Use Python 3.11 or later for this feature. Existing TTS use still supports 3.10.

```sh
uv pip install 'livekit-plugins-rime[realtime]'
```

The realtime extra installs `rimelabs-sdk>=0.1.0a14` from PyPI. For source
development, install this plugin with its realtime extra from the repository root:

```sh
uv sync --package livekit-plugins-rime --extra realtime --inexact
```

```python
import os
from livekit.agents import Agent, AgentSession
from livekit.plugins import rime

session = AgentSession(
    llm=rime.realtime.RealtimeModel(
        endpoint=os.environ["PRISM_URL"],
        voice=os.environ["PRISM_VOICE"],
    ),
)
await session.start(agent=Agent(instructions="Give short, clear answers."), room=ctx.room)
session.generate_reply(user_input="Hello!")
```

Set `RIME_API_KEY` and configure the production Prism endpoint:

```sh
export PRISM_URL=wss://prism.api.rime.ai/v1/realtime
```

The endpoint must accept the SDK's Bearer API-key handshake. Local development
also accepts `ws://localhost/...`. You can pass `client=Rime(...)` to reuse credentials; the
plugin does not close an injected client. Create one model per AgentSession.

The adapter initializes instructions, tools, and text history before it sends
input. `set_input_text` identifies a typed turn; `update_chat_ctx` appends history
or tool results. No timer determines whether text is a history note or a turn.
LiveKit's `playback_finished` hook reports the response ID, playback position,
and stop cause, including skipped audio. Caller speech sends a truncate.
Application stops send a playback receipt. Late transcripts keep the user's
original position in history.

Configuration changes, agent handoff, AgentTask, tool progress updates, video,
manual audio commit, forced tool selection, per-reply tools, and disabling turn
detection are unsupported. These operations emit an unrecoverable model error.
Prism cannot enforce `tool_choice="none"`. LiveKit requests it when the tool-step
limit is reached or while draining a tool reply; the session fails explicitly
at that point instead of starting another tool round. Use a separate TTS model
for `session.say`.

Microphone input and output streams have bounded queues. A queue overflow ends
the session. LiveKit drains unused audio when audio output is absent or disabled.
There is no automatic reconnect or tool replay in the SDK or Prism adapter.
An application can opt into LiveKit's realtime fallback adapter. It forwards
initialization, typed turns, and playback reports. An explicit restart opens a
new conversation with supported user/assistant text history; it cannot restore
a pending tool round or replay tool history. Local configuration records are
excluded from that history. Concurrent use of one Prism model remains unsupported.

Each fallback generation carries its playback session. Reports reach that
session even after a replacement, without relying on globally unique provider
IDs or retaining an ID lookup table.
Initialization preserves saved settings when a caller omits them. Explicit empty
instructions or tools replace the saved values. If a reply is accepted just
before wrapper cancellation, the wrapper interrupts its owning session and
reports skipped playback.

An SDK `RealtimeAdmissionTimeout` fails only the reply request. It does not close
the conversation. Unknown request outcomes still fail the session.
An SDK `ResponseAbandoned` notice identifies a late accepted reply whose caller
was cancelled. The adapter discards its unused output and reports zero playback
once the terminal response supplies a drain token.

The local tests exercise the SDK through a WebSocket peer and AgentSession;
they do not establish production endpoint compatibility.

Protocol and mapping reference: rimelabs/sglang-omni at
`bc8a500249536f369d20bc094baa9169e5986758`, Apache-2.0.
