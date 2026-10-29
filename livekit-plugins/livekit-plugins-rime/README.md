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

## Prism realtime

Prism provides speech input, reasoning, tool calls, and speech output through the
Python Rime SDK. Use Python 3.11 or later. TTS still supports Python 3.10.

```sh
uv pip install 'livekit-plugins-rime[realtime]'
```

The realtime extra requires `rimelabs-sdk>=0.1.0a25`. For source development:

```sh
uv sync --package livekit-plugins-rime --extra realtime --inexact
```

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import rime
from rimelabs_sdk.realtime import TurnDetection

model = rime.realtime.RealtimeModel(
    turn_detection=TurnDetection(eagerness="low", end_of_turn_timeout_ms=5000),
)
session = AgentSession(llm=model)
await session.start(agent=Agent(instructions="Give short, clear answers."), room=ctx.room)
session.generate_reply(user_input="Hello!")
```

Set `RIME_API_KEY`. The default endpoint is `wss://prism.api.rime.ai/v1/realtime`.
Use `endpoint=` for a custom deployment and `voice=` if it has no default voice.
The endpoint must accept the SDK's Bearer API-key handshake. Local endpoints can
use `ws://localhost/...`. An injected `client=Rime(...)` remains caller-owned.
Create one model per AgentSession and close it when the application finishes.

Use the ordinary `livekit.agents.AgentSession` and
`livekit.agents.llm.RealtimeModelFallbackAdapter`. The plugin has no replacement
session, activity, fallback class, or custom audio output.

```python
from livekit.agents.llm import RealtimeModelFallbackAdapter

model = RealtimeModelFallbackAdapter([
    rime.realtime.RealtimeModel(),
    rime.realtime.RealtimeModel(),
])
session = AgentSession(llm=model)
```

### LiveKit compatibility

Prism targets LiveKit's standard realtime interface from 1.7.1 onward. We test
1.7.1, 1.8.6, and 1.8.8 with `rimelabs-sdk==0.1.0a24`. The package now requires
`livekit-agents>=1.8.9`; installation on older versions needs a release that permits it.

LiveKit 1.7.1 fixes active-response cancellation through `SpeechHandle.interrupt()`.
Versions 1.7.1 through 1.8.5 still ignore realtime `max_tool_steps`, including for
OpenAI. Use [1.8.6 or later](https://github.com/livekit/agents/pull/7555) if you need
that limit. Explicit `tool_choice="none"` works on 1.7.1.

Run the source compatibility check from the repository root:

```sh
uv run --no-project --python 3.11 \
  --with 'livekit-agents[codecs]==1.7.1' \
  scripts/test_rime_compatibility.py --core-version 1.7.1
```

Substitute both version values to test another release. The runner verifies import
paths and runs all Rime regressions locally without API keys. Three tool-limit
cases are expected failures before 1.8.6; other failures or skips fail the check.
Package installation and live services need separate validation.

### Turns and interruption

Microphone input uses Prism's server turn detection. The adapter forwards
speech events and uses LiveKit's standard cancellation and message truncation.
It requests interruption on caller speech. Early speech hints do not change
playback volume. There is no custom per-response interruption policy.

The adapter sends no playback completion receipts. Prism estimates playback
completion from its audio timing plus 1.5 seconds of slack. After caller
interruption it waits up to 300 ms for a precise position, then uses its estimate.
Skipped or muted output can therefore delay the next request until the estimate
expires. LiveKit supplies a truncation position when it interrupts a partially
played message, including application stops.

Text passed to `generate_reply(user_input=...)` is inserted into history before
a proactive reply request. It is not submitted through Prism's dedicated typed
turn operation. The response can use that history but does not identify the text
as its current input item. Per-reply instructions can accompany this request.

### Tools and settings

Tools execute through LiveKit. The adapter submits each result and requests a
tool continuation. Completed tool history can be loaded into a fresh session;
resuming a pending tool call in a new session is unsupported.

`tool_choice="auto"` uses the registered tools. `tool_choice="none"` is emulated:
the adapter clears the server's tools, waits for acknowledgment, requests the
reply, then restores the tools after response creation. A cancelled request must
settle before restoration. An unexpected tool call from a suppressed response
fails the session before LiveKit can execute it. A failed restoration also fails
the session because its configuration is uncertain.

These operations are not atomic. An automatic speech response that starts during
the temporary change can also receive no tools. This applies at LiveKit's tool-step
limit and during handoff when LiveKit requests a final reply without tools.
When Prism supplies a settings revision, each response keeps the tool policy of
that revision, even if it arrives after a later settings update.
Required or named tool choices remain unsupported. The plugin does not advertise
native per-response tool selection. Rejecting these choices leaves the session open.

Use `agent.update_instructions()` and `agent.update_tools()` for agent settings.
Updates wait for Prism acknowledgments. Handoff uses the normal LiveKit sequence;
instructions, history, and tools are separate updates. There is no atomic handoff.
Use `await model.update_options(voice="...")` for the voice or
`await model.update_options(response_metadata={"step": "2"})` for response labels.
For one explicit reply on a direct Prism `RealtimeSession`, call
`await session.generate_reply(instructions="Check in", metadata={"step": "3"})`.
The per-reply labels replace matching session labels. Replies that Prism starts from
caller speech use session labels.
When one call changes both settings and response labels, it sends two separate
requests. If the label update fails, the earlier accepted settings remain applied.
A successful return confirms that Prism acknowledged all updates in the call.

Tool fillers are optional. Pass
`tool_fillers={"lookup": PrismFiller(type="text", text="Checking your order.")}`,
with `PrismFiller` imported from `rimelabs_sdk.realtime`. The keys must name
registered tools. Use `type="none"` to suppress filler or `type="auto"` for generated
filler. Change them with `model.update_options(tool_fillers=...)`.

### History and recovery

`agent.update_chat_ctx()` supports user, assistant, and system text messages and
completed tool calls/results. Each imported message has at most 4000 characters.
An imported result must immediately follow its call. Images, video, developer
messages, and tool progress are unsupported. LiveKit configuration and handoff
records stay local. Importing history does not execute tools or start a reply.

Append and deletion preserve retained turns. Replacing or reordering history
rebuilds completed items with new server IDs. A rebuild requires no active reply
or pending tool call. Prism has no atomic history replacement: replies can see
intermediate context between operations. An explicit event-scoped refusal during
an append preserves acknowledged items and reports the original provider error.
Keep the requested context to retry or correct the unaccepted items. Preserve
newer conversation items and accepted item IDs when retrying. The plugin does not retry
automatically or execute tools again. Replies requested through the plugin are
blocked until rejected tool results have been submitted. Server-initiated speech
replies are still controlled by Prism.

An unknown request outcome, session failure, or partially applied deletion or
replacement still ends the session. Standard fallback replays the current agent
context, which can already contain the requested edit even when Prism rejected it.

Set `truncation="auto"`, `truncation="disabled"`, or
`truncation=RetentionRatio(0.9)` at construction or through `model.update_options()`.
Import `RetentionRatio` from `rimelabs_sdk.realtime`. Automatic truncation retains
0.8 of the available history capacity. Disabling it reports an error when the
context no longer fits. This setting requires deployment support and does not
erase the conversation record. Omit it for older deployments.

The SDK and adapter do not automatically reconnect or replay accepted requests.
Standard LiveKit fallback can restart with supported history. It cannot restore
an active tool round. Set `regenerate_on_swap=False` when an automatic replacement
reply is unwanted. Text input submitted again is another history insertion.
Unknown request outcomes fail the SDK session; they are not retried.

This plugin uses unchanged LiveKit core. Core records requested agent history
before the provider acknowledges it, so rejected or cancelled edits can leave
agent and provider history different. Fallback startup can also return before a
backup is ready, and a failed restart can emit more than one reconnect event.
Use a known valid history when starting a replacement session after these errors;
the plugin does not add its own history rollback or startup failover logic.

### Events and limits

The Prism `RealtimeSession` emits `rime_server_event_received` with each SDK
`SessionEvent`, before the plugin processes it. The event contains `session_id`,
`event_id`, `request_id`, and a typed `payload`. The `request_id` correlates an event
with a client request; it is not the provider request ID set at connection time.
Read `ResponseStarted` payloads for response labels, input item IDs, settings
revisions, and saved instruction IDs. Read `SpeechStopped.end_reason` for the
speech end reason. Standard LiveKit speech, transcription, and generation events
remain available on the session.

The SDK handles settings acknowledgments internally and does not expose them in
this event stream. Await `model.update_options()` to confirm a settings update.
There is no separate `session_updated` event.

Read `ItemTruncated` and `ResponseEnded` from `event.payload`.
`ResponseEnded.usage` has token counts when Prism reports them. Available retained
text updates the plugin's history only while the server item ID still matches. `kept_text=None`
means unavailable; an empty string means no words remain. This text describes
server history, not necessarily what was heard, and does not replace LiveKit's
playback transcript. Acknowledgments during generation may have no retained text,
with no later update guaranteed.

`ResponseEnded.truncation` reports items excluded from that reply's prompt.
It does not delete local history. `None` means no exclusions were reported or the
information was unavailable. Read `items_left_out` and `first_kept_item_id` from
the summary for diagnostics.

`PrismError` retains the provider fault, code, scope, IDs, and `retryable`.
`PrismModelError` also exposes retryability. Retryability describes whether an
unchanged request may succeed; recoverability describes whether the session can
continue. Neither starts an automatic retry. An admission timeout fails the reply
without closing an otherwise valid session.

Manual audio commit, disabling server turn detection, per-reply tools, and video
are unsupported. Use a separate TTS model for `session.say()`.
Audio and output queues are bounded; overflow fails the session. The adapter
retains up to 128 generations, evicting completed generations as new ones arrive.
Truncation is available only for retained output references.

Local WebSocket peer tests cover the SDK and ordinary AgentSession. They do not
establish production endpoint compatibility or model quality for history-based
text replies.

Run the Rime regression tests from the repository root:

```sh
uv run pytest tests/test_plugin_rime.py --unit
```

The tests include response settings revisions, completed tool history rebuilds,
history snapshot isolation, and cancellation during acknowledged updates.
Validate typed replies and audio playback against the target Prism deployment
separately. Caller-interruption checks require a speech fixture with nonzero audio.
