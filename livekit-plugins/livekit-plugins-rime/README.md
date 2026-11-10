# Rime plugin for LiveKit Agents

Support for voice synthesis and speech recognition with [Rime](https://rime.ai/).

See [https://docs.livekit.io/agents/integrations/tts/rime/](https://docs.livekit.io/agents/integrations/tts/rime/) for more information.

## Installation

TTS supports Python 3.10 or newer. STT requires Python 3.11 or newer, LiveKit Agents
1.8.10 or newer, and the optional `stt` extra.

```bash
pip install livekit-plugins-rime
```

## Pre-requisites

You'll need an API key from Rime. It can be set as an environment variable: `RIME_API_KEY`

## Speech recognition

STT uses the Rime Python SDK and requires Python 3.11 or newer. Install the `stt` extra:

```bash
pip install 'livekit-plugins-rime[stt]'
```

The extra installs `livekit-agents>=1.8.10` and `rimelabs-sdk>=0.1.0a29`, which provides
unary transcription, revised-hypothesis streaming, and append-fragment streaming.
STT requires core fixes for non-retryable recognition errors. This minimum does not
apply to the `realtime` extra, which supports LiveKit Agents 1.7.1 or newer.
On Python 3.10, the SDK dependency is skipped even when the extra is requested;
upgrade Python to use STT. TTS does not require the SDK.

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

Append-fragment support is available in the published `rimelabs-sdk>=0.1.0a29`
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

## Prism realtime

Prism provides speech input, reasoning, tool calls, and speech output through the
Python Rime SDK. Use Python 3.11 or later. TTS still supports Python 3.10.

```sh
uv pip install 'livekit-plugins-rime[realtime]'
```

The realtime extra requires `rimelabs-sdk>=0.1.0a29`. For source development:

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

The plugin package requires `livekit-agents>=1.7.1`. Prism requires Python 3.11
or later and the `realtime` extra. Earlier installation checks covered LiveKit
1.7.1 and 1.8.0 through 1.8.10 with `rimelabs-sdk==0.1.0a28`.
These checks apply to the PR build. Future package releases can raise the minimum
LiveKit version; check their declared requirements before upgrading.

Keep the application's LiveKit version pinned when adding the plugin.
Install the plugin from the PR commit into a Python 3.11+ virtual
environment. Replace `<commit>` with its full commit ID:

```sh
uv pip install --no-sources 'livekit-agents==1.7.1' \
  'livekit-plugins-rime[realtime] @ git+https://github.com/naszzz/agents.git@<commit>#subdirectory=livekit-plugins/livekit-plugins-rime'
```

For Git installs, [`--no-sources`](https://docs.astral.sh/uv/concepts/dependencies/#disabling-sources)
ignores the monorepo's development source overrides. Dependency resolution still
checks the package requirements and uses the published core version you select.
Without this flag, uv selects the core from the same Git workspace instead.

LiveKit 1.7.1 fixes active-response cancellation through `SpeechHandle.interrupt()`.
Versions 1.7.1 through 1.8.5 still ignore realtime `max_tool_steps`, including for
OpenAI. Use [1.8.6 or later](https://github.com/livekit/agents/pull/7555) if you need
that limit. Explicit `tool_choice="none"` works on 1.7.1.

### Turns and interruption

Microphone input uses Prism's server turn detection. The adapter forwards
speech events and uses LiveKit's standard cancellation and message truncation.
It requests interruption on caller speech. Early speech hints do not change
playback volume. There is no custom per-response interruption policy.

Choose playback behavior when you create the model:

| Option | Behavior |
| --- | --- |
| `playback="auto"` (default) | Use Prism estimates for normal completion. No output wrapper is required. |
| `playback="estimated"` | Use Prism estimates for normal completion. |
| `playback="discard"` | Discard audio before frame allocation and output buffering. Report zero played milliseconds at response end. |

All modes retain interruption reporting. Discard keeps text and tools active. It
keeps the audio capability and an empty audio stream, so LiveKit does not invoke
an automatic TTS fallback. Explicit application `say()` calls are separate.
Discard still uses server synthesis, audio transfer, and SDK parsing. It is not
text-only server generation. Speed depends on the deployment and workload.

Normal playback uses Prism's estimates. No application setup is required for
RoomIO or caller-owned audio outputs. LiveKit still reports the played position
when interrupting a reply through the standard message-truncation interface.
Measured normal-completion reporting is not part of the public plugin API.

Text passed to `generate_reply(user_input=...)` is inserted into history before
a proactive reply request. It is not submitted through Prism's dedicated typed
turn operation. The response can use that history but does not identify the text
as its current input item. Per-reply instructions can accompany this request.

### Tools and settings

Tools execute through LiveKit. The adapter submits each result and requests a
tool continuation. Completed tool history can be loaded into a fresh session;
resuming a pending tool call in a new session is unsupported.

If a sub-task or newer turn has superseded the tool continuation, the adapter
requests a reply using the recorded result and the agent's instructions. That
reply can speak or call another tool; it is not restricted to summarizing the result.

Tool errors retain the application's message and are marked as failures. The
adapter does not add a retry policy or execute a failed tool again itself.
With `tool_choice="auto"`, the model chooses whether to speak or call a tool.
Applications that require another collection attempt should control that workflow.

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
Use `model.update_options(voice="...")` for the voice or
`model.update_options(response_metadata={"step": "2"})` for response labels.
For one explicit reply on a direct Prism `RealtimeSession`, call
`await session.generate_reply(instructions="Check in", metadata={"step": "3"})`.
The per-reply labels replace matching session labels. Replies that Prism starts from
caller speech use session labels.
When one call changes both settings and response labels, it sends two separate
requests. If the label update fails, the earlier accepted settings remain applied.
`model.update_options()` and `RealtimeSession.update_options()` return immediately.
Updates are queued in order; stored options change after acknowledgment. A later
application reply waits for queued options. Provider-initiated speech uses the
settings in effect when Prism starts it. Recoverable update refusals emit an
`error` event and preserve the session and last accepted settings. Connection or
session failures retain their fatal handling. Options set before the first
session opens become its initial configuration.

Tool fillers are optional. Pass
`tool_fillers={"lookup": PrismFiller(type="text", text="Checking your order.")}`,
with `PrismFiller` imported from `rimelabs_sdk.realtime`. Each key names a tool.
The plugin sends fillers only for active tools and keeps the configured map
across handoffs. Use `type="none"` to suppress filler or `type="auto"` for generated
filler. Replace the map with `model.update_options(tool_fillers=...)`, or pass
`tool_fillers={}` to clear it.

### History and recovery

`agent.update_chat_ctx()` supports user, assistant, and system text messages and
completed tool calls/results. Each imported message has at most 4000 characters.
An imported result must immediately follow its call. Images, video, developer
messages are unsupported. LiveKit configuration and handoff
records stay local. Importing history does not execute tools or start a reply.

Tools can use `await ctx.update(...)` to report progress. The first update completes
the provider's original tool call. LiveKit imports later updates and the final
result as completed call/result pairs and requests replies when needed. These
imports do not execute the tool again. `silent=True` records progress without
requesting a reply.

The default `context_mode="shared"` keeps one conversation through agent and task
handoffs. Ordinary context updates are views of that conversation. Empty or reduced
views preserve omitted history and the original pending tool call. A task can
change instructions and tools without removing that call. Recorded generated,
played, and retained transcript variants do not force a rebuild. Unknown changes
or insertions are rejected. LiveKit's separate warning about tasks on realtime
models remains; automatic task-isolated history is not supported.

For headless tests, discard audio output:

```python
model = rime.realtime.RealtimeModel(playback="discard")
```

Shared mode also preserves omitted messages that were never played. To replace
or clear provider history through `agent.update_chat_ctx()`, set
`context_mode="strict"`. Reducing local context in shared mode does not remove
that history from Prism.

Explicit `context_mode="strict"` retains replacement semantics. Append, prepend,
middle insertion, deletion, replacement, and reorder are supported. Edits that
need a rebuild require no active reply or pending call. Prism deletes a completed
call and its result together. Every write has a separate acknowledgment;
replacement is not atomic. Use shared mode for tasks inside tools, where LiveKit
can omit the parent's pending call from the task's context.

`PrismHistoryUpdateError` is an `Exception` with the following fields. Catch this
concrete type for history update errors. The base class does not inherit from
`llm.RealtimeError`. A partial ordinary update that only appends live tool results
uses a subclass of both types if an accepted result has `reply_required=True`.
LiveKit can then request a reply from that accepted result. The update still
raises a structured error and logs a warning. All other failed updates
stop their dependent work, including the reply to rejected typed input.

| Field | Meaning |
| --- | --- |
| `status` | `rejected`, `partial`, or `uncertain` |
| `recoverable` | The known provider state supports later independent operations |
| `confirmed_chat_ctx` | Detached projection, or `None` for an uncertain outcome |
| `needs_reconciliation` | Local context may differ; optional reconciliation can repair it |

Error codes are `history_update_rejected`, `history_update_partial`, and
`history_update_uncertain`. Error messages exclude transcript content. The plugin records each accepted write before the next write. A known
partial edit keeps the connection open. An acknowledgment timeout or unknown
write outcome retains the SDK's terminal failure behavior. No mutation is retried.
If new history items arrive during a write, the plugin makes at most one history
read to establish their order. If active-item omissions prevent confirmed order,
the outcome is uncertain. Missing items are never treated as deletions.

Both context modes allow independent replies and updates after a local validation
failure, explicit provider rejection, or known partial update. Each later update
starts from confirmed provider history. Strict mode still applies an exact target;
shared mode still preserves omitted items and rejects conflicting edits.

Local and provider history can differ after a failed update. The error includes
the confirmed context when the outcome is known. Later updates start from
confirmed provider history.

A refused typed input does not start
its own reply. A later `session.generate_reply()` can start an independent reply
without repair. Accepted tool results retain their continuation ownership and are
not submitted or executed again. A known partial tool-result-only batch can
continue automatically from an accepted result that requires a reply. Fully
rejected batches, mixed text/tool edits, explicit replacements or clears, and
uncertain writes still stop dependent replies. The application can submit missing
results or request a later independent continuation for an accepted result.
Provider-initiated speech remains under Prism's control.

The realtime extra requires SDK alpha29 or later. The plugin uses `hold_response`
for long tasks and delayed playback beyond 128 responses.
The SDK retains up to 128 ordinary and 128 protected responses. Active generation,
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
fallback swaps retain their existing dropped-update and startup behavior.

### Response metrics

Each provider `ResponseEnded` event produces one standard LiveKit
`RealtimeModelMetrics` event. AgentSession forwards it through `metrics_collected`
and includes its token counts in session usage. Metrics include the response ID,
provider token counts when available, generation duration, and cancellation status.
Local interruption waits for the provider's final event so final usage is retained.
If the connection ends without that event, no completed-response metric is emitted.

`ttft` measures time from response creation to the first nonempty audio chunk
received, including in discard mode. It is `-1` if no audio arrives. Duration
ends at the provider's final response event. These timings exclude reply admission
and device playback; they do not measure end-to-end conversational latency.
Elapsed times use a monotonic clock.

Missing usage and unavailable audio, text, and cache token breakdowns use
LiveKit's zero defaults. A zero therefore does not prove that the provider used
no tokens. Tokens per second uses the reported output tokens and generation duration.

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
this event stream. Returning from `update_options()` does not confirm acceptance.
There is no separate `session_updated` event. Update refusals use the standard
`error` event.

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

Manual audio commits, per-reply tools, and video inputs log a warning and are
ignored. Disabling server turn detection, opening a second concurrent session,
and invalid tool choices raise `llm.RealtimeError` without closing an active
session. Use a separate TTS model for `session.say()`.
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
