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

Prism combines speech input, reasoning, tool calls, and speech output in one
realtime model. The plugin calls the Python Rime SDK for all Prism protocol work.
Use Python 3.11 or later for this feature. Existing TTS use still supports 3.10.

```sh
uv pip install 'livekit-plugins-rime[realtime]'
```

The realtime extra installs `rimelabs-sdk>=0.1.0a22` from PyPI. For source
development, install this plugin with its realtime extra from the repository root:

```sh
uv sync --package livekit-plugins-rime --extra realtime --inexact
```

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import rime
from rimelabs_sdk.realtime import TurnDetection

model = rime.realtime.RealtimeModel(
    turn_detection=TurnDetection(eagerness="low", end_of_turn_timeout_ms=5000),
    response_metadata={"workflow": "support"},
)
session = AgentSession(
    llm=model,
)
await session.start(agent=Agent(instructions="Give short, clear answers."), room=ctx.room)
session.generate_reply(user_input="Hello!")
```

Set `RIME_API_KEY`. The default endpoint is `wss://prism.api.rime.ai/v1/realtime`.
Set `endpoint=` for a custom deployment and `voice=` if it has no default voice.
For a complete room agent, see `examples/voice_agents/rime_prism.py`.

Optional constructor settings include `turn_detection`, `timeouts` (`RealtimeTimeouts`),
`interrupt_on_speech`, `request_id`, and `response_metadata`. Turn timing can be set
only when opening the conversation. Omitted values use the SDK or server defaults.

The endpoint must accept the SDK's Bearer API-key handshake. Local development
also accepts `ws://localhost/...`. You can pass `client=Rime(...)` to reuse credentials; the
plugin does not close an injected client. Create one model per AgentSession.

The adapter initializes instructions, tools, and supported history before it sends
input. `set_input_text` identifies a typed turn; `update_chat_ctx` synchronizes history
or submits live tool results. No timer determines whether text is a history note or a turn.
LiveKit's `playback_finished` hook reports the response ID, playback position,
and stop cause, including skipped audio. Caller speech sends a truncate.
Application stops send a playback receipt. Late transcripts keep the user's
original position in history.

### Session updates and handoff

Use `agent.update_instructions()` and `agent.update_tools()` to change the current
agent. The adapter waits for Prism to acknowledge the settings. Instruction and
tool changes during handoff are sent together. A rejected update leaves the
acknowledged settings unchanged and does not end the conversation. Cancellation
waits for an outstanding acknowledgment, so the local settings reflect accepted
updates even when the caller cancels.

Use `await model.update_session(voice="...", interrupt_on_speech=False)` for
provider settings, or `await model.update_session(response_metadata={"step": "2"})`
to replace labels for future replies. `model.session_info` exposes the current
acknowledged settings and the request ID Prism actually used. A gateway may
replace a supplied `request_id`.

Handoff and AgentTask entry/return can reuse the conversation when both agents
use the same model and preserve equivalent conversation history. Instructions
and tools can change. Completed tool history can be loaded into a fresh
conversation. Resuming a pending tool call in a fresh conversation remains unsupported. The SDK keeps the settings revision that
each response started with, including its interruption policy.

### History and context limits

Use `await agent.update_chat_ctx(context)` to append or remove history and to
replace or reorder completed history. Supported items are user, assistant, and
system text messages, plus completed tool calls and results. History text can
contain at most 4000 characters per message. Imported tool calls require results;
imported results must immediately follow their calls. Importing history never
starts a reply or executes a tool. Images, video, developer messages, and tool
progress remain unsupported. LiveKit configuration records stay local.

`await model.get_history()` returns the SDK's immutable tuple of completed server
history items. These items carry server IDs. LiveKit retains its own item IDs and
conversation record. Running replies and their inputs can be absent from the
server snapshot; their absence does not delete local history.

Removing either half of a completed tool pair requires removing both from the
requested context. An active call must still receive its result through LiveKit.
Append and deletion updates preserve retained turns. A change to content or order
rebuilds completed history with fresh server IDs, so it requires no active replies
or pending tool calls and costs requests proportional to the history size.

History operations require deployment support. A rejected update before any
change leaves the conversation usable. If an update fails after an earlier
operation committed, the adapter reports a fatal error; an explicit fallback can
restore the prior context in a new conversation. The adapter does not replay
requests with unknown outcomes. A cancelled update waits for acknowledgment and
keeps accepted changes. Agent context is committed only after acceptance, while
preserving conversation items received during the update.

Prism does not provide atomic history replacement. An existing reply keeps its
original context, and a reply that starts between separate operations can see an
intermediate context. A new reply during a rebuild stops that update; if any
changes committed, the conversation requires recovery.

For long conversations, pass `truncation="auto"`, `truncation="disabled"`, or
`truncation=RetentionRatio(0.9)` to the constructor. Import `RetentionRatio` from
`rimelabs_sdk.realtime`. Use `await model.update_session(truncation="auto")` to
change the policy during a session. `auto` keeps 0.8 of the available history
capacity when truncation is needed. `disabled` reports `context_length_exceeded`
when context no longer fits. A retention ratio must be above zero and at most one.
Instructions and tools remain. Truncation does not erase the conversation record.

The plugin omits this setting unless supplied, preserving existing requests for
older servers. The deployment must support the setting before it is enabled.
`model.session_info.truncation` reports the acknowledged policy, or `None` if the
server does not publish it. Accepted policies survive explicit restarts. This
context setting is separate from reporting interrupted audio playback.

### Tool fillers

Pass `tool_fillers={"lookup": PrismFiller(type="text", text="Checking your order.")}`
to the model constructor, importing `PrismFiller` from `rimelabs_sdk.realtime`.
The keys must name registered tools. `type="none"` suppresses filler; `type="auto"`
uses generated filler and accepts optional `instructions`. Omitted settings use
the server default. Replace settings with
`await model.update_session(tool_fillers={...})`. Removing a tool also removes
its filler setting. The adapter still executes each tool through LiveKit.

### Events, errors, and early speech hints

Subscribe to the model's `generation_created`, `session_updated`, `speech_stopped`,
and `speech_hint` events. `PrismGenerationCreatedEvent` extends LiveKit's generation
event with reply labels in `metadata`,
`input_item_ids`, `settings_revision`, and `provider_request_id`. A stopped-speech
event is a `PrismSpeechStoppedEvent` with `end_reason`. Import these types from
`livekit.plugins.rime.realtime`. Interim transcripts are forwarded before the final
transcript. Fallback preserves generation metadata and its owning playback session.
Prism diagnostics remain on these events; shared generation traces do not copy
the provider request ID, input item IDs, settings revision, or speech end reason.
LiveKit's existing response ID attribution remains unchanged.

`PrismError` retains the original `fault`, `code`, `scope`, request and response IDs,
and `retryable`. `PrismModelError` extends LiveKit's model error event with the
same retry classification. It is also available from `livekit.plugins.rime.realtime`.
`retryable` describes whether an unchanged request may succeed again;
`recoverable` describes whether the current LiveKit session can continue.
Neither flag starts an automatic retry. Do not replay accepted typed turns or
tool execution. A context-length error requires shorter context. Configure history truncation,
remove supported history items, or start a new conversation with shorter history. An unknown request outcome still closes the SDK conversation.

A speech hint may be a false alarm. LiveKit lowers playback, without cancellation,
on outputs that support gain control. `speech_hint_gain=0.25` and
`speech_hint_timeout=0.6` control the gain multiplier and restoration timeout.
Repeated hints extend the timeout. A later application gain change is preserved;
a subsequent hint uses that gain as its baseline. Confirmed speech follows the interruption
policy of the reply being played. Console and room output support gain; custom
outputs can opt in with `AudioOutputCapabilities(gain=True, ...)` and apply their
gain when submitting samples. Device or remote audio already submitted can retain
its previous gain. Playback duration and receipts are unchanged.

### Remaining limits

Tool progress updates, video, manual audio commit, forced tool selection,
per-reply tools, and disabling turn detection remain unsupported.
Prism cannot enforce `tool_choice="none"`. LiveKit requests it when the tool-step
limit is reached or while draining a tool reply; the session fails explicitly
at that point instead of starting another tool round. Use a separate TTS model
for `session.say`.

Microphone input and output streams have bounded queues. A queue overflow ends
the session. Reaching the 128-generation retention limit with unresolved playback
also ends the session, rather than forgetting a required receipt. LiveKit drains
unused audio when audio output is absent or disabled.
There is no automatic reconnect or tool replay in the SDK or Prism adapter.
An application can opt into LiveKit's realtime fallback adapter. It forwards
initialization, typed turns, and playback reports. An explicit restart opens a
new conversation with supported text and completed tool history; it cannot restore
a pending tool round or rerun tools from that history. Local configuration records are
excluded from that history. Concurrent use of one Prism model remains unsupported.

Each fallback generation carries its playback session. Reports reach that
session even after a replacement, without relying on globally unique provider
IDs or retaining an ID lookup table.
Initialization preserves saved settings when a caller omits them. Explicit empty
instructions or tools replace the saved values. If a reply is accepted just
before wrapper cancellation, the wrapper interrupts its owning session and
reports skipped playback.

An SDK `RealtimeAdmissionTimeout` fails only the reply request. It does not close
the conversation. A typed turn refused before acceptance remains staged for a
later explicit `generate_reply()` call; the adapter does not retry it automatically.
A new explicit typed input can replace that unsent turn. Accepted or cancelled
requests are not restaged. Unknown request outcomes still fail the session.
An SDK `ResponseAbandoned` notice identifies a late accepted reply whose caller
was cancelled. The adapter discards its unused output and reports zero playback
once the terminal response supplies a drain token.

The local tests exercise the SDK through a WebSocket peer and AgentSession;
they do not establish production endpoint compatibility.

Protocol and mapping reference: rimelabs/sglang-omni at
`bc8a500249536f369d20bc094baa9169e5986758`, Apache-2.0.
