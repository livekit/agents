# Duplex Model Abstraction — Design

Status: **proposed**. Branch `longc/duplex-model`, based on `main` (`bbf163fe1`).

Supersedes the experiment on `longc/openai-live-realtime`, which implemented GPT-Live directly as a
`RealtimeModel`. That branch works and is the reference for protocol behaviour; this document
re-derives the framework integration from `main`.

## Goal

Give full-duplex speech models a first-class home in the framework, without forking the voice
pipeline and without a type that lies about what it supports.

Concretely, v1 ships:

- a `DuplexModel` / `DuplexSession` interface that plugin authors implement, carrying no method the
  model cannot perform and publishing no name we are not ready to keep,
- one framework-owned adapter that segments a duplex model's continuous output stream and presents
  it to `AgentActivity` as an ordinary `RealtimeSession`,
- GPT-Live (Responses delegation, WebSocket) reshaped onto that interface.

## Background: what full duplex actually changes

A half-duplex realtime model produces audio only while it holds the turn. A full-duplex model runs
both directions continuously: it decides for itself when to speak, self-manages barge-in, and — the
part that matters here — **emits output audio unconditionally, including silence between turns**.

The landscape as of 2026-08:

| Model | Access | Turn events | Tools / delegation |
|---|---|---|---|
| OpenAI `gpt-live-1` | alpha API | yes (`turn.created` / `turn.done`) | yes (Responses or client) |
| NVIDIA PersonaPlex-7B | open weights, no hosted API | not exposed | no |
| Moshi / Kyutai (Gradium) | open weights; Gradium sells STT+TTS | none | no |
| DuplexOmni, Voila, VITA-Audio, SALMONN-Omni, Step-Audio R1.1 | research | — | — |
| Gemini Live, Nova 2 Sonic, Qwen-Audio-3.0-Realtime, xAI Voice | GA | turn-scoped | — (half duplex) |

**One hosted duplex API exists.** That fact drives most of the scoping decisions below: the design
must earn its keep against GPT-Live alone, and stay cheap to reverse.

## The one structural gap

Everything else full duplex offers is already expressible:

| Duplex property | Expressible on `main` today? |
|---|---|
| Model keeps talking through user speech | capability flag (`server_barge_in`, added here) |
| No client interrupt / truncate / commit | capability flags guarding no-ops |
| Model decides when to speak, or stays silent | server-initiated `generation_created`, supported |
| Reasoning delegation | orthogonal; parked (see Non-goals) |
| Intonation-aware input | model-internal, no framework surface |
| **Audio the model emits with no transcript** | **nothing can express this** |

Every path to `session.output.audio` in `agent_activity.py` runs through a speech task holding a
`SpeechHandle`. There is no lane for audio that plays but is not a turn. Backchannels, breath,
laughter, and non-lexical vocalisations therefore cannot reach the room.

On the experiment branch they are dropped outright: inter-turn audio is held in a ~5 s rolling
buffer and only the frames overlapping the *next* transcript onset survive
(`live_realtime_model.py:813-821`).

Closing that gap is the value of this work. It is one missing output lane, which is why the design
below is an adapter rather than a second model hierarchy.

### Why the framework must not classify

A tempting shortcut is "audio without transcript is a backchannel". It isn't: singing, laughter, a
sound effect, and a non-verbal answer are all untranscribed audio that are emphatically not asides.
Any classifier the framework applies will be wrong for some of them.

So the framework never classifies. Two independent questions:

- **What reaches the room** — decided by an energy gate (mandatory; see below).
- **What enters the chat context** — decided by the transcript.

Music with no transcript plays and is not recorded. A backchannel plays and is not recorded. If the
model does transcribe them, they are recorded. The framework does not need to tell them apart.

## Non-goals

**Always-on forwarding is rejected.** GPT-Live's audio never stops
(`live_realtime_model.py:54`: *"keyed on transcript activity, audio never stops"*) — ~100 ms frames
arrive unconditionally, silence included. Publishing all of it means the sink never idles:
`clear_buffer()` on interruption refills instantly, `drain()` never sees an idle output,
`agent_state` must ignore playback entirely, and the room receives permanent room tone. That is a
second lifecycle for the whole voice pipeline, in exchange for content that is silence.

**No shared base class for `RealtimeModel` and `DuplexModel`.** Considered and rejected — see
Alternatives.

**Client delegation is parked.** In Responses mode the backend's tool calls arrive as
`response.function_call_arguments.done` and flow through the generation's existing `function_stream`;
the framework's tool executor runs them, `_update_chat_ctx` returns the results as
`delegation.function_call_output.create`, and `auto_tool_reply_generation=True` covers the
continuation. That is entirely existing machinery. `DelegationCreatedEvent`,
`push_delegation_result`, `client_delegation`, `AgentSession(delegation_llm=)`,
`Agent.delegation_node` and `_execute_delegation_tools` — all present on
`longc/agent-delegation` and `longc/openai-live-realtime` — stay parked.

**Video, WebRTC and SIP transports** are out of scope. WebSocket only; `push_video` is a no-op.

## Architecture

```
plugin                       framework (shared)                    unchanged
────────────────────         ──────────────────────────────        ─────────────────
DuplexModel          ──►     DuplexRealtimeAdapter(RealtimeModel)
DuplexSession        ──►     _DuplexRealtimeSession(RealtimeSession)  ──►  AgentActivity
  audio_stream                 energy gate + segmenter                     (no changes
  transcript/turn events       generation assembly                          beyond 2 flags)
  tools, chat_ctx              dead methods, once
```

`AgentActivity` only ever sees a `RealtimeSession`. The plugin only ever implements `DuplexSession`.

### `livekit-agents/livekit/agents/llm/duplex.py` (new)

```python
@dataclass
class DuplexAudioFrame:
    frame: rtc.AudioFrame
    message_id: str | None   # the model's turn id, None while the plugin does not know it
    start_ms: int | None     # position on the model's own timeline


@dataclass
class DuplexTranscriptDelta:
    text: str
    message_id: str | None
    start_ms: int | None
    end_ms: int | None       # end of the span covered, on the model's timeline


class DuplexModel:
    """A speech model that listens and speaks concurrently.

    Its audio streams continuously whether or not it is speaking, and barge-in is its own.
    """

    @property
    def capabilities(self) -> DuplexCapabilities: ...
    def session(self) -> DuplexSession: ...
    async def aclose(self) -> None: ...


class DuplexSession(ABC, rtc.EventEmitter[DuplexEventTypes]):
    @property
    def audio_stream(self) -> AsyncIterable[DuplexAudioFrame]: ...

    chat_ctx, tools          # read-only views
    push_audio, push_video
    aclose

    # the framework's contract with the plugin, not an app-facing API — see Public API surface
    _update_instructions / _update_chat_ctx / _update_tools / _update_options
    _update_session          # all three at once, right after session()
    _generate_reply          # only where capabilities.manual_response_creation

    # events
    #   transcript_delta            DuplexTranscriptDelta
    #   turn_started / turn_ended   assistant turn boundary, by message_id
    #   input_speech_started / input_speech_stopped
    #   input_audio_transcription_completed
    #   function_call               llm.FunctionCall
    #   metrics_collected / error / session_reconnected
```

No `interrupt`, `truncate`, `commit_audio`, `clear_audio`, `say` or `start_user_activity` — no
duplex model can do them. A plugin author cannot implement a method the model does not have.

`_update_session` hands instructions, chat context and tools over as one unit, mirroring
`RealtimeSession._update_session`. It exists because a model whose configuration is immutable once
the session starts — GPT-Live's voice `instructions` are — has to compose its first outbound event
from the complete configuration, and would otherwise have to infer completeness from the order in
which the individual updates happen to arrive.

`DuplexCapabilities` carries only what varies between duplex providers: `user_transcription`,
`auto_tool_reply_generation`, `mutable_tools`, `mutable_chat_context`, `mutable_instructions`, and
`manual_response_creation`. Barge-in and the absence of truncation are properties of the type.

`manual_response_creation` is on that list rather than absent, because "the client cannot ask the
model to speak" is a fact about GPT-Live, not about duplex: nothing in listening and speaking
concurrently prevents a model accepting a prompt, and a proactive greeting is the most-missed thing
without it. It defaults to False and `_generate_reply` raises, so a model whose protocol allows it
simply overrides the method.

### `livekit-agents/livekit/agents/llm/duplex_adapter.py` (new)

Mirrors the existing `RealtimeModelFallbackAdapter` / `_FallbackRealtimeSession` pattern.

**One id space.** `message_id` on frames, transcript deltas and turn events is always the model's
turn id. Never a per-fragment id: the framework reads a change of id as a change of turn, so mixing
the two splits one turn into fragments — which is exactly what shipped `" Sure,"`, `" I"`,
`" can do"` as three chat items until the GPT-Live plugin stopped falling back to `item.id`. A
plugin that does not yet know the turn reports None.

`DuplexRealtimeAdapter(RealtimeModel)` reports:

```python
RealtimeCapabilities(
    audio_output=True, turn_detection=True, server_barge_in=True,
    message_truncation=False, supports_say=False,
    manual_function_calls=False, per_response_tool_choice=False,
    user_transcription=..., mutable_tools=..., mutable_chat_context=...,
    mutable_instructions=..., auto_tool_reply_generation=...,
    manual_response_creation=...,
)
```

`_DuplexRealtimeSession(RealtimeSession)` forwards the shared surface verbatim, implements
`interrupt` / `truncate` / `commit_audio` / `clear_audio` / `say` as the documented no-ops, forwards
its `generate_reply` to a model that supports it, and does the real work: the gate and the
segmenter.

The dead methods exist in exactly one place, in framework code nobody subclasses. The API plugin
authors write against stays honest.

## Public API surface

Honest about what a duplex model can do is not yet honest about *how*. At n=1 hosted models, every
state-changing method on `DuplexSession` is named after a framework concept and shaped by one
protocol, and the fit is already visibly bad. GPT-Live's chat-context hook diffs the incoming
context by item id to recover what is new, then fans the result out to two unrelated protocol
events — `delegation.function_call_output.create` and `session.context.append`. The name promises
replacement of a context the model cannot replace; the diffing exists only to undo the
whole-context shape the framework imposed. Freezing that into the abstraction commits the next
duplex model to a workaround for this one.

So v1 publishes only what a second model cannot plausibly contradict:

| on `DuplexSession` | why |
|---|---|
| `audio_stream`, `push_audio`, `push_video`, `aclose`, `chat_ctx`, `tools` | public — true of any model that listens and speaks, and the read-only views promise nothing |
| `_update_instructions`, `_update_chat_ctx`, `_update_tools`, `_update_options`, `_generate_reply`, `_update_session` | underscored — the framework's contract with the plugin, free to rename once a second protocol says what the shape should be |

The underscore is not a claim that apps should never touch state. It is a claim that the *framework*
should not yet be the one naming those operations. Apps reach the real thing through the plugin,
where the API can be protocol-shaped and truthful:

```python
class GPTLiveSession(llm.DuplexSession):
    def append_context(self, text: str) -> None: ...              # session.context.append
    def send_delegation_output(self, *, call_id, output) -> None: ...
    def send_event(self, event: dict[str, Any]) -> None: ...      # escape hatch

    async def _update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        ...  # diffs, then calls the two above
```

The framework hook implemented in terms of the public methods is the check that the public methods
are shaped right: if the framework's needs can be expressed through them, an app's can too.

### Reaching the session

The adapter is an implementation detail and must not appear in application code. Two things make
that hold.

**The plugin exports the duplex model itself**, not a wrapper. `Agent` and `AgentSession` already
wrap a `DuplexModel` on the way in, so a factory returning `DuplexRealtimeAdapter(...)` only hides
the session from the person who constructed the model.

**`Agent.duplex_session`**, alongside the existing `realtime_llm_session`, unwraps
`_DuplexRealtimeSession.duplex_session` and raises when the agent is not running on a duplex model:

```python
async def on_enter(self) -> None:
    live = cast(openai.GPTLiveSession, self.duplex_session)
    live.append_context("The caller is a premium customer.")
    live.on("openai_server_event_received", self._trace)
```

The `cast` is the price of not committing to a shared API, and it is the same one an app already
pays to reach any provider-specific realtime surface.

One consequence to document rather than fix: context appended this way never reaches
`session.history`, because it never went through the framework. Bypassing the abstraction means
bypassing what the abstraction maintains.

## The energy gate

A model that emits unconditionally needs *something* to decide what is not silence. This is
mandatory infrastructure, not a tuning knob.

**Not a VAD.** Reusing the session's VAD abstraction was considered and rejected:

1. It drops exactly what we are trying to save. Silero is a speech/non-speech classifier; breath,
   laughter, sighs, humming and singing are the untranscribed paralinguistics this whole design
   exists to preserve.
2. It adds onset latency to every turn — activation window plus `min_speech_duration` is ~100 ms+
   in front of every agent utterance.
3. It makes a VAD a hard dependency. Duplex sessions run without one by design.

**An adaptive noise floor instead**, so there is no absolute, model-specific threshold:

```
noise_floor  = quietest frame of the last 10 seconds
open   when  energy > noise_floor * k_open
close  when  energy < noise_floor * k_close  sustained for hangover
```

Three properties make one set of defaults port across providers:

- **The floor is a rolling minimum, not a percentile.** Speech cannot drag a minimum upward, because
  even continuous delivery leaves gaps between words. A percentile over a window that fills with
  speech converges on speech level, and since phoneme-to-phoneme energy varies by more than the open
  ratio, the gate would flap or wedge shut during a long monologue.
- **Every duration is measured in audio, not wall clock.** The window is 10 seconds of audio rather
  than a frame count, so a provider sending 20 ms frames and one sending 100 ms frames behave
  identically, and network jitter cannot move a boundary.
- **`hangover` is grounded in speech, not in a provider.** Pauses inside an utterance run to ~400 ms
  and boundaries between utterances start around 500 ms, so 0.5 s separates utterances without
  splitting one. This is a property of human-like speech, so it holds for any model producing it.

`k_open` / `k_close` are dimensionless (hysteresis ~2:1). All of them are defaults on a pluggable
policy, so a model with different output characteristics can substitute its own.

A frame the model itself attributes to a turn is passed to the gate as `forced`: it is always
output, and it holds the gate open so the turn's hangover starts from its last tagged frame. That
is what lets the untagged tail of a turn survive a lagging boundary, and it removes the gate's
dependence on having heard silence first — a session that opens straight into speech never
calibrates a floor, and without this the tail would be dropped.

The gate also reports whether each frame carried sound *ignoring* forcing. The segmenter needs that
separately: forced frames are forwarded whether or not they are audible, so only the unforced
verdict says where the speech actually ended.

## The segmenter

Consumes `audio_stream` and produces ordinary `GenerationCreatedEvent`s. It is a **source
demultiplexer, not a sink writer** — it never touches `session.output.audio`, so there is no second
writer, no ordering hazard, and `SpeechHandle`, `_schedule_speech`, `drain()` and `agent_state` are
structurally untouched.

```
for frame in audio_stream:
    voiced = gate.update(frame)

    if voiced and idle:
        open a burst  ->  emit generation_created

    if in a burst:
        push frame to the burst's audio stream

    if not voiced for hangover and in a burst and no turn is open:
        the sound has stopped -> decide whether the burst may close
```

### When a burst closes

The gate says when the *sound* stopped. It cannot say whether the model still owes transcript for
it, and closing early strands the rest: the late fragment opens a burst of its own, which never
receives audio, so the gate cannot close that one either and the *next* turn's audio gets adopted
into it. Observed as one turn's final `"."` prefixing the next turn's chat item.

Once the sound has stopped, the burst closes on the first of:

| | condition | why it is safe |
|---|---|---|
| 1 | `turn_ended` for this burst's turn | the model's events and its transcript share one stream, so every fragment of turn T is on the wire before `turn.done(T)`. Ordering, not a guess |
| 2 | transcript coverage has reached the burst's **voiced** audio | both sides are model-timeline milliseconds, so the answer does not move with network latency |
| 3 | a liveness timeout (3 s) | neither of the above can ever arrive |

None of them can cut audio short: all three additionally require the gate to have heard the sound
stop, and while a turn is open its frames are tagged, which holds the gate open.

Rule 2 measures **voiced** audio deliberately. GPT-Live tags trailing silence into a turn — 2.2 s of
it in one measured case, between the last word at 26200 ms and `turn.done` at 28400 ms — so
comparing against everything forwarded made the watermark permanently unsatisfiable and every turn
fell through to the timeout. Silence is never transcribed, so it must not count.

Rule 3 is a liveness bound rather than a tuning knob, and is unreachable for any model that reports
turns. Only a model that starts transcribing and then stops mid-turn can get there.

A burst with no transcript at all is a backchannel and owes nothing — unless the model has claimed
it as a turn, in which case its text simply has not started yet. Turn identity is the discriminator;
at the moment the gate closes the two are otherwise indistinguishable.

Attribution:

- `message_id` changing on a frame forces a burst boundary, so the model's own turn structure wins
  wherever it provides one.
- Transcript deltas for a burst's `message_id` become the generation's `text_stream`, as
  `TimedString`s. The segmenter knows exactly which frames it forwarded and their `start_ms`, so it
  maps the model's timeline onto the forwarded playback timeline. The transcription synchroniser
  then paces text against real speech rather than an estimated rate.
- A burst with no transcript still plays. It produces no chat item, because
  `agent_activity.py:4063` already skips message creation for empty `forwarded_text` — no framework
  change needed. **This is the backchannel.**
- A fragment arriving with no burst open is classified by its span. One reaching **past** the audio
  already forwarded leads a turn about to start, and the burst it opens is adopted by the audio that
  follows — the normal path, silent. One that does not has outlived its own audio: it is emitted as
  its own generation with an error log, since losing transcript is worse than an odd chat item, and
  it is marked orphaned so no later turn can be adopted into it. Classifying by span rather than by
  "has this burst any audio" matters: the cruder test also caught leading fragments, and would have
  split the opening words off every turn for any model that tags its first audio frame.
- A reconnect closes the open burst and forgets any open turn. A dropped connection never delivers
  the turn's end, which would otherwise hold the burst open for the rest of the session. Doing this
  in the adapter rather than each plugin makes it a guarantee instead of a thing to remember.

## Framework changes

Three changes, all additive and defaulted, so every existing realtime plugin is unaffected. One is
designed but deferred.

### 1. `server_barge_in` (required)

```python
# llm/realtime.py — RealtimeCapabilities
server_barge_in: bool = False
"""Whether the model handles barge-in itself, stopping its own output when the user speaks."""
```

Without it, `_on_input_speech_started` (`agent_activity.py:1891`) interrupts unconditionally, on the
Realtime API's guarantee that server VAD already cancelled the response server-side. For a duplex
model nothing was cancelled, so cutting playout on a turn event cuts real speech — on the experiment
branch a backchannel produced a hard audio cut and split one utterance across two assistant
messages.

With the flag set:

- `_on_input_speech_started` keeps the user-state and audio-recognition updates and leaves playback
  alone; the assistant turn ends on its own boundary.
- `_interrupt_by_audio_activity` (`agent_activity.py:1984`) drops its `turn_detection` early return,
  so an explicitly configured VAD or interruption detector can still cut the agent off. A *default*
  VAD stays unwired.
- `allow_interruptions=False` is no longer rejected, since nothing interrupts from the model's event.

Client-side interruption remains opt-in and lossy by nature: nothing can be sent to stop the model
and its context cannot be truncated, so it believes it said what the user never heard.

### 2. `manual_response_creation` (required)

```python
# llm/realtime.py — RealtimeCapabilities
manual_response_creation: bool = True
"""Whether generate_reply() can ask the model to speak, rather than only waiting for it to"""
```

Defaults True, which is the behaviour of every shipped realtime model. The adapter mirrors the
duplex model's own flag. When it is False, `AgentActivity._generate_reply` returns a completed
`SpeechHandle` and logs at debug rather than dispatching a call that can only fail — asking a model
that decides for itself when to speak is a no-op, not an error. That is what makes the greeting path
quiet instead of error-logging once per session.

### 3. `carries_transcript` (deferred, conditional)

```python
# llm/realtime.py — GenerationCreatedEvent
carries_transcript: bool = True
```

`_on_first_frame` (`agent_activity.py:3799`) fires on the first *audio* frame regardless of
transcript and unconditionally does three things. For an untranscribed burst:

| effect | verdict |
|---|---|
| `_on_start_of_agent_speech()` — echo-suppression window | **correct**, the agent is vocalising |
| `_disable_vad_interruption_soon()` | bounded: `_restore_interruption_by_audio_activity()` runs when the generation finishes, so suppression lasts only the ~300 ms burst |
| `_update_agent_state("speaking")` | a state flap and an `AgentStateChangedEvent` per backchannel |

None is a correctness bug. The real cost is observable churn: a `speech_created` event, an
`agent_turn` tracing span, a state flap and an interruption enable/disable cycle for every breath.

This field can only be set at burst onset, which requires an onset-time signal from the protocol —
see Open questions. **Ship v1 without it**, measure the churn against a real session, and add it
only if the telemetry noise justifies it.

## GPT-Live plugin

`livekit-plugins-openai/.../realtime/gpt_live_model.py` reshapes onto `DuplexSession`. The
protocol handling — `session.update` composition, both delegation modes, `input_audio.append`,
usage deltas, reconnect — carries over from the experiment branch unchanged.

What is deleted, because the framework now owns it:

- `_pending_audio`, `_MAX_PENDING_AUDIO_FRAMES`, `_flush_pending_audio`, `_feed_audio`
- `_close_current_generation` / `_closing_turn_end_ms` turn-carving and the two-timeline correlation
- `_GENERATION_IDLE_TIMEOUT`, which existed only because audio was not a usable liveness signal
- the `item.id` fallback when labelling a transcript, which mixed per-fragment ids into the turn id
  space and split every leading fragment into its own chat item

Roughly 150 lines of the plugin's most fragile code. In their place the plugin emits every
`output_audio.delta` as a `DuplexAudioFrame` tagged with the open turn's `message_id` (or `None`),
and forwards transcript and turn events.

The plugin exports the duplex model directly — `AgentSession(llm=)` and `Agent(llm=)` accept one, so
a wrapper would only put the adapter between the app and the session it constructed:

```python
class GPTLiveModel(llm.DuplexModel):
    def session(self) -> GPTLiveSession: ...
```

`GPTLive` rather than `Live`: inside LiveKit the bare word carries no information, and `Realtime`
already names the other OpenAI API in this same plugin (`openai.realtime.RealtimeModel`). The API is
alpha, so the rename is free now and expensive later.

### Syncing the chat context

`_update_chat_ctx` diffs the incoming context against what has already gone out with
`llm.utils.compute_chat_ctx_diff`, the same machinery the Realtime plugin uses, and routes what is
new by kind. A result answering a call **this connection delegated** goes back on
`delegation.function_call_output.create`, the backend model's channel, and is not context at all.
Everything else becomes **one** `session.context.append` carrying a `role: text` transcript of what
was added since the last sync.

Which side a tool call falls on is tracked, not assumed. A resumed call, an agent handoff and a
reconnect all hand over prior `function_call_output` items whose `call_id` the backend never issued
— and `_reset_for_reconnect` clears the mirror, so after a drop the entire history re-syncs. Sent
blind, every one of those becomes a delegation output answering nothing. The session therefore
records the calls it delegates, clears them with the connection, and renders the rest into the
transcript as `tool call:` / `tool result:` lines, where they read as history instead of as protocol.

One append rather than one per message, because the Live API's context entries carry no role: split
across events, a user question and the answer to it arrive as unattributed fragments. Written as a
transcript in a single block they stay legible, which also answers how to seed a resumed call —
flattening prior history into one text block is not a workaround here, it is the shape the protocol
has.

A removal or a revision is logged as an error and otherwise ignored. The model keeps what it has
been told, and pretending otherwise would leave the plugin's mirror disagreeing with the session.

A known limitation disappears: *"`turn.done`(assistant) lags the turn's last audio delta, so the
filler streamed in between is forwarded as part of the segment"* is no longer a defect, because
inter-turn audio is supposed to be forwarded — it is simply untranscribed.

Two protocol facts the plugin relies on, both confirmed against live sessions:

- `output_transcript.added` can arrive **before** the audio it describes, not only after. The
  segmenter classifies a fragment by its span rather than assuming it trails.
- `turn.created`(assistant) arrives ~450 ms after the turn's first transcript fragment, and
  `turn.delta` mirrors `output_transcript.added` with the stable turn id but ~700 ms later. The
  plugin reads the earlier, unlabelled source because delaying every fragment would defeat the
  `TimedString` pacing; the framework absorbs the missing label.

## Alternatives considered

**Ship GPT-Live as a plain `RealtimeModel` (the experiment branch).** Works today. Rejected because
backchannels are unreachable, and the turn-carving that makes it work is per-plugin,
protocol-specific and would be reimplemented differently by the next duplex model.

**`DuplexModel(RealtimeModel)` by inheritance.** Zero framework churn, but reproduces the problem:
`generate_reply` is typed to return a future and `agent_activity` awaits it, so duplex must raise and
callers must defend against a method the type promises. The turn-based control sites become runtime-
guarded rather than type-guarded, and mypy — strict in this repo — cannot help.

**A shared `RealtimeModelBase` with `RealtimeModel` and `DuplexModel` as siblings.** The most
type-honest option, and the one carried furthest in design. Rejected on cost: ~34 shared call sites
in `agent_activity` need renaming, `RealtimeCapabilities` and `MessageGeneration` both split,
`AgentSession(llm=)` typing widens, the fallback adapter needs a decision, and
`RealtimeModelBase` next to `RealtimeModel` is a standing trap for plugin authors. That is a large,
hard-to-reverse change against a framework delta that is otherwise one capability flag, at n=1
hosted models.

The adapter achieves the actual goal — the API people write against has no dead methods — and
confines the dishonesty to one internal class.

**Revisit trigger:** a second duplex model whose control surface `RealtimeSession` cannot express
(for example an explicit "yield the floor" command), or client delegation landing and forcing
`delegation_created` / `push_delegation_result` / `client_delegation` onto `RealtimeSession` where no
turn-based model uses them. Either is the point to reconsider the base class — with two models'
worth of evidence instead of one.

## Open questions

1. **Do assistant turn events cover untranscribed vocalisations?** Partly answered: `turn.created`
   *does* carry `role: "assistant"`, but its `start_ms` lags the true speech onset, which is why the
   experiment opens generations from the first `output_transcript.added` instead. Turn events
   therefore cannot be the sole boundary source, and the energy gate is required for onset precision
   regardless. What remains unknown is whether a backchannel with no transcript produces an
   assistant turn at all; if it does, `carries_transcript` becomes knowable at onset. Answerable
   from one session's event log.
2. **`wait_for_playout()` returns `__last_playback_ev`, not the event matching the caller's target**
   (`io.py:221-235`). Harmless today because segments are serialised through one speech task. Not
   changed by this design, but worth noting if concurrent waiters ever appear.
3. **Should a backchannel be interruptible?** It currently would be, as an ordinary `SpeechHandle`.
   Probably right, and moot at ~300 ms.
4. **Duplex events can overtake the audio they follow.** `audio_stream` is a channel drained by the
   segmenter task, but `transcript_delta` / `turn_started` / `turn_ended` are emitted synchronously
   by the plugin, so an event fired after a frame was queued is processed before it. The transcript's
   real lag hides this today, but any audio backlog would land a fragment in the previous burst. The
   fix is to funnel the events through the same channel as the audio so one task sees one ordered
   stream.
5. **An orphaned fragment still becomes a chat item.** One that outlived its audio is emitted as a
   generation with nothing behind it, surfacing as a standalone `"."`. Deliberate — losing
   transcript is worse — and now rare and loudly logged rather than silent. Whether such a burst
   should produce a chat item at all remains open, and the same applies to a function-call-only
   burst.
6. **An open turn still forwards silence.** Forced frames are output whether or not they are
   audible, so a model that announces a turn and then pauses inflates the generation's audio and
   skews `playback_time` for later fragments. The watermark no longer suffers — it measures voiced
   audio — but playout pacing still does. Dropping the silence would fix the timing and break the
   pacing, so this needs a real decision rather than a tweak.
7. **Whether one accessor is enough.** `Agent.duplex_session` covers provider-specific events and
   methods, but an app holding only an `AgentSession` has to go through `current_agent` first, and
   the session does not exist until the activity starts — so subscribing to a plugin's own events
   (GPT-Live's `openai_server_event_received`) means doing it from `on_enter`. Both are true of
   `realtime_llm_session` today. Worth revisiting only if apps hit it.

## Testing

- `tests/test_duplex_adapter.py` (unit, 33): the gate against synthetic frames — silence, steady
  room tone, speech over tone, hysteresis, hangover, a rolling-minimum floor that sustained speech
  cannot raise, and identical decisions at 20 ms and 100 ms frames; segmenter boundaries from
  `message_id` changes and from silence; tagged audio forwarded through a mid-turn pause; a tail
  surviving `turn_ended`; a turn adopting a burst already open; `turn_started` closing a foreign
  burst; transcript fragments never splitting a burst; a trailing fragment landing in the burst the
  gate just closed and the next turn not inheriting it; leading versus orphaned classification and
  the error log on the orphan only; the watermark closing a turn without waiting; `turn_ended`
  releasing a turn whose transcript never caught up; transcript→playback timeline mapping;
  reconnect and shutdown leaving no timer pending; configuration handed over as one unit;
  `_generate_reply` rejected by a model that cannot be asked and forwarded by one that can; a duplex
  model wrapped on its way into `AgentSession` and `Agent`; `Agent.duplex_session` returning the
  plugin's own session rather than the adapter's, and raising for a model that is not duplex.
- `tests/test_transcript_sync_timed.py` (unit, 2): a timed fragment releases its trailing word
  without waiting for a delimiter the model only sends with its next turn.
- `tests/test_realtime_barge_in.py` (unit, 6): `server_barge_in` gating — the speech-started event
  leaves playback alone with the flag and still interrupts without it, an explicitly provided VAD
  still interrupts either way, and the `allow_interruptions=False` rule.
- `tests/test_plugin_openai_live_realtime.py` — **not yet written.** About half the experiment
  branch's version tested the deleted turn-carving internals; the rest (`_build_live_url`,
  `_to_tool_choice`, `session.update` composition, `_update_chat_ctx` →
  `delegation.function_call_output.create`, usage deltas, user-turn mapping) ports over largely
  unchanged.
- Manual: `python examples/voice_agents/openai_gpt_live.py console` against the alpha endpoint —
  confirming backchannels are audible, transcripts stay aligned, and no hard audio cut occurs on
  user speech.
