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

The realtime extra requires `rimelabs-sdk>=0.1.0a26`. For source development:

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
session, activity, or fallback class. It provides an optional audio output wrapper.

```python
from livekit.agents.llm import RealtimeModelFallbackAdapter

model = RealtimeModelFallbackAdapter([
    rime.realtime.RealtimeModel(),
    rime.realtime.RealtimeModel(),
])
session = AgentSession(llm=model)
```

### LiveKit compatibility

The plugin package requires `livekit-agents>=1.8.10`. The Prism adapter also has
source compatibility checks against LiveKit 1.8.10 with
`rimelabs-sdk==0.1.0a26`. These checks load the plugin source directly; they do not
install this package on older core versions.

LiveKit 1.7.1 fixes active-response cancellation through `SpeechHandle.interrupt()`.
Versions 1.7.1 through 1.8.5 still ignore realtime `max_tool_steps`, including for
OpenAI. Use [1.8.6 or later](https://github.com/livekit/agents/pull/7555) if you need
that limit. Explicit `tool_choice="none"` works on 1.7.1.

Run the source compatibility check from the repository root:

```sh
uv run --no-project --python 3.11 \
  --with 'livekit-agents[codecs]==1.8.10' \
  scripts/test_rime_compatibility.py --core-version 1.8.10
```

Substitute both version values to test another release. The runner verifies import
paths and runs all Rime regressions locally without API keys. Three tool-limit
cases are expected failures before 1.8.6; other failures or skips fail the check.
Package installation and live services need separate validation. Add
`--with-editable /path/to/rime-sdk/python` to the uv command and
`--sdk-source /path/to/rime-sdk/python` to the script arguments to test the
companion SDK source. Run both Rime test files with the current source core too.

### Turns and interruption

Microphone input uses Prism's server turn detection. The adapter forwards
speech events and uses LiveKit's standard cancellation and message truncation.
It requests interruption on caller speech. Early speech hints do not change
playback volume. There is no custom per-response interruption policy.

Choose playback behavior when you create the model:

| Option | Behavior |
| --- | --- |
| `playback="auto"` (default) | Report measured completion for output identified by the installed wrapper. Use Prism estimates elsewhere. |
| `playback="estimated"` | Use Prism estimates for normal completion. |
| `playback="discard"` | Discard audio before frame allocation and output buffering. Report zero played milliseconds at response end. |

All modes retain interruption reporting. Discard keeps text and tools active. It
keeps the audio capability and an empty audio stream, so LiveKit does not invoke
an automatic TTS fallback. Explicit application `say()` calls are separate.
Discard still uses server synthesis, audio transfer, and SDK parsing. It is not
text-only server generation. Speed depends on the deployment and workload.

For a caller-owned sink, install the wrapper before starting the agent:

```python
model = rime.realtime.RealtimeModel(context_mode="shared")
session = AgentSession(llm=model)
session.output.audio = model.wrap_audio_output(my_audio_sink)
await session.start(Agent(instructions="Help the caller."))
```

For RoomIO, install it at the start of `on_enter`, before requesting an opening
reply. LiveKit sets up RoomIO before it calls `on_enter`:

```python
class Assistant(Agent):
    async def on_enter(self):
        sink = self.session.output.audio
        if sink is not None:
            self.session.output.audio = model.wrap_audio_output(sink)
        self.session.generate_reply(instructions="Greet the caller.")

session = AgentSession(llm=model)
await session.start(Assistant(instructions="Help the caller."), room=ctx.room)
```

The wrapper reads frame metadata before downstream resampling. Local protocol
checks cover the normal AgentSession audio path, 24 kHz and 48 kHz sinks, multiple
messages, and synchronous completion events. The RoomIO example is tested with
real RoomIO and an RTC audio source, using local publication and subscription
substitutes. Actual room transport and device playback need a deployment check. Custom audio nodes must preserve the frame metadata and sample sequence.
Missing metadata, mixed output, discarded frames, replacement sinks, and missing
completion events keep estimates in effect. Application TTS and `say()` audio
cannot release a Prism response. A bare `playback_finished` listener cannot prove
which response finished.

After generation ends, an unconfirmed output releases its local hold after the
generated audio duration plus `timeouts.request_s`. This cleanup never reports
completion. Paused or delayed output can thus return to estimated completion.
Late events after close cannot report to another provider session.

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

The default `context_mode="strict"` retains replacement semantics. Append,
prepend, middle insertion, deletion, replacement, and reorder are supported.
Edits that need a rebuild require no active reply or pending call. Prism deletes
a completed call and its result together. Every write has a separate
acknowledgment; replacement is not atomic.

For headless tests and tasks inside tools, use:

```python
model = rime.realtime.RealtimeModel(playback="discard", context_mode="shared")
```

Shared mode treats ordinary context updates as views. Empty or reduced views
preserve omitted history and the original pending tool call. A task can change
instructions and tools without removing that call. Recorded generated, played,
and retained transcript variants do not force a rebuild. Unknown changes or
insertions are rejected. LiveKit's separate warning about tasks on realtime
models remains; automatic task-isolated history is not supported.

Shared mode also preserves omitted messages that were never played. Use an
explicit edit to remove those messages. Task views and provider history need not
be equal. Exact edits, including punctuation changes, use these helpers:

```python
await model.replace_chat_ctx(session.current_agent, target_context)
await model.clear_chat_ctx(session.current_agent)
await model.reconcile_chat_ctx(session.current_agent)
```

Helpers require an active agent that uses this model directly. Pause competing
local context edits and handoffs during each call. Helpers preserve local
configuration records and use `exclude_invalid_function_calls=False` so a task's
smaller tool list cannot strip the parent call. Pending calls cannot be cleared.
These helpers do not support a model inside a fallback adapter.

`PrismHistoryUpdateError` is a `llm.RealtimeError` with the following fields:

| Field | Meaning |
| --- | --- |
| `status` | `rejected`, `partial`, or `uncertain` |
| `recoverable` | The known provider state can still be used after reconciliation |
| `confirmed_chat_ctx` | Detached projection, or `None` for an uncertain outcome |
| `needs_reconciliation` | Local context still needs repair |

Error codes are `history_update_rejected`, `history_update_partial`, and
`history_update_uncertain`. Messages and notice representations exclude transcript
content. The plugin records each accepted write before the next write. A known
partial edit keeps the connection open. An acknowledgment timeout or unknown
write outcome retains the SDK's terminal failure behavior. No mutation is retried.

The explicit helper attempts local repair after a known failure, then raises the
original error. Check `needs_reconciliation`; an event that races with repair can
prevent it. For ordinary core-owned updates, install a synchronous notice callback
and arrange recovery in application code:

```python
import asyncio
from livekit.plugins.rime.realtime import PrismHistoryUpdateError

notices = asyncio.Queue()
model = rime.realtime.RealtimeModel(
    context_mode="shared", on_history_update=notices.put_nowait,
)
session = AgentSession(llm=model)

async def recover_next_update():
    notice = await notices.get()
    if notice.recoverable and notice.needs_reconciliation:
        # Application must first pause competing edits and handoffs.
        await model.reconcile_chat_ctx(session.current_agent)

try:
    await model.replace_chat_ctx(session.current_agent, target_context)
except PrismHistoryUpdateError as error:
    if error.recoverable and error.needs_reconciliation:
        await model.reconcile_chat_ctx(session.current_agent)
    raise
```

Until reconciliation succeeds, dependent client replies and ordinary edits are
blocked. Results for already-live calls can still settle. Accepted results retain
their continuation ownership; they are not executed again. Reconciliation resumes a blocked tool continuation once. Other client replies
need a new request through the session. A refused typed input cannot silently
start a reply without that input. Provider-initiated speech remains under Prism's
control. The helpers do not rewrite the session's past event log.

SDK alpha26 supports these features within its existing reference-cache lifetime.
Long tasks and delayed playback beyond 128 responses require the SDK version with
`hold_response` (currently the companion SDK source change, not a published floor).
That SDK retains up to 128 ordinary and 128 protected responses. Active generation,
unresolved calls, and holds count toward the protected bound. Limit exhaustion
raises an error instead of evicting a pending owner. No audio is retained by holds.

Set `truncation="auto"`, `truncation="disabled"`, or
`truncation=RetentionRatio(0.9)` at construction or through `model.update_options()`.
Import `RetentionRatio` from `rimelabs_sdk.realtime`. Automatic truncation retains
0.8 of the available history capacity. Disabling it reports an error when the
context no longer fits. This setting requires deployment support and does not
erase the conversation record. Omit it for older deployments.

The adapter retries initial HTTP refusals with status 408, 429, 500, 502, 503, or
504 using the structured `status_code` supplied by SDK `0.1.0a26` or later. Other HTTP errors,
unknown connection failures, and initialization timeouts fail without retry.
A server that mislabels a billing refusal as 503 will receive bounded retries;
the adapter cannot infer billing state from that status alone.

Set `conn_options=APIConnectOptions(max_retry=3, retry_interval=2, timeout=10)`
to configure retries and the maximum duration of each connection attempt,
including SDK initialization. Import `APIConnectOptions` from `livekit.agents`.
The default delays are 0.1, 2, and 2 seconds. A valid SDK `retry_after_s` is a
minimum delay. `startup_timeout=30` bounds all attempts and delays together.
These limits bound startup failure reporting; closing waits for SDK cleanup.
If the server's requested delay cannot fit, the original refusal ends startup.
The SDK's `timeouts=` still controls individual protocol operations. Set
`max_retry=0` to disable retries. Closing the session cancels startup and delays.
Errors emitted while a retry will occur have `recoverable=True`; permanent or
exhausted failures emit `recoverable=False`. Waiting settings and reply requests
remain pending until initialization succeeds or fails.

The SDK and adapter do not automatically reconnect an established conversation
or replay accepted requests.
Standard LiveKit fallback can restart with supported history. It cannot restore
an active tool round. Set `regenerate_on_swap=False` when an automatic replacement
reply is unwanted. Text input submitted again is another history insertion.
Unknown request outcomes fail the SDK session; they are not retried.

This plugin uses unchanged LiveKit core. Core records requested agent history
before the provider acknowledges it, so rejected or cancelled edits can leave
agent and provider history different. Fallback startup can also return before a
backup is ready, and a failed restart can emit more than one reconnect event.
Use a known valid history when starting a replacement session after these errors;
use the direct-model helpers for known history failures. Fallback swaps retain
their existing dropped-update and startup behavior.

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
continue. Startup retries follow the connection policy above; these flags do not
retry a failed reply. An admission timeout fails the reply
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
