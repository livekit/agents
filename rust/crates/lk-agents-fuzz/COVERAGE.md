# Voice driver coverage

What `lk-agents-fuzz` covers of the Rust voice driver (`lk-agents-voice`), and what it does not.
Every number is measured, not claimed: a 1,024-seed survey and `cargo llvm-cov` over the same
suite, at `edc90fe24` with the working tree of 2026-10-23. Written by hand for now; section 12
says how to re-measure and what makes it generated.

Scope: the driver only. The Python voice pipeline, threads and lanes, the A2A door and the real
providers are outside this suite (section 11).

## 1. What the framework is for

A person talks with an agent, in real time, by voice. A developer writes the agent against an
API (an `Agent` with instructions, tools and models, an activity that runs it, handoffs to
other agents, tasks, hooks) and may call that API at any moment of the conversation: while the
person speaks, while the model thinks, while the agent talks, while a tool runs, during a
handoff, while the session drains. Underneath, several models of different kinds work at once:
something that hears (STT, VAD, end-of-turn, interruption detector, AMD, IVR), something that
thinks (an LLM, or a realtime model that also hears and speaks), something that speaks (TTS),
and outputs that play, pause and report back. Between thinking and speaking sits the speech
scheduler: one speech on the floor at a time, the rest queued, gated by authorization, floor
holds, the gap between speeches and drains.

Five promises follow. Everything here is measured against them.

| | Promise | Broken looks like |
|---|---|---|
| P1 | **The conversation keeps moving.** Every API call returns; every speech ends; the session closes when asked. | dead air forever, a hung `await`, a session that never closes |
| P2 | **Turn-taking is right.** The agent answers when the person is done, one speech at a time, stops when barged in on (if allowed), plays nothing it was not authorised to. | talking over the person, two replies at once, answering mid-sentence, ignoring a finished turn |
| P3 | **What is said is what was meant.** Audio played is what was synthesised, transcripts are the words, every frame the person said reaches the models listening, history records what was heard. | lost or duplicated words, a transcript that does not match the audio |
| P4 | **The developer stays in control, at any time.** An API call takes effect or is refused explicitly; handoffs, updates and tools apply to the right work; replaced things stay replaced. | an update applied to the wrong reply, a tool run twice, an old output still written to |
| P5 | **Failures are survivable and accounted for.** A failing model is retried, surfaced or closes the session; everything opened is closed. | a silent failure, a leaked stream, a session that dies without telling the host |

## 2. The search

One generator (`src/search/conversation.rs`). A run is a setup (which models exist and how
they are configured) and a sequence of inputs, on paused virtual time in tokio's fixed task order.
The generator plays four actors, each of which acts only on what it observed:

- **The person** speaks, pauses, hesitates, makes noise, has someone else in the room, drops off
  the call and comes back. While the agent speaks they listen: they may talk over it anywhere in
  the reply, from its first word to its last, and they say "mm-hm" now and then without wanting
  the floor, which the fake interruption detector judges as a real one would. When no reply comes
  they wait as long as their patience lasts, then speak again. Their speech reaches the models as
  microphone frames and verdicts: a recogniser transcribes only audio the driver forwarded to it.
- **The developer's program** calls the API between turns and, beat by beat, inside them:
  while the person speaks, while the agent thinks, speaks or runs a tool. It answers hooks and
  tool calls.
- **The end of the call** is its own dial: an average call length drawn per seed, from 30 s to
  two hours, flat over orders of magnitude. When it comes, the developer closes, closes with a
  drain, or drains then closes, or the person hangs up, or the room closes; the developer keeps
  calling the API while the session winds down. Most runs outlast their budget; about one in
  eight ends early this way.
- **The providers** (models, outputs, room) answer what they were asked, at their own speed,
  and fail in the ways real ones do.
- **Randomness**: before each move the conversation makes, a run may first make uniform moves --
  any legal row of the table except one that ends the call, no actor behind it, labelled in the
  case (`Case::random`). How
  many is drawn per seed (section 7), from almost never (a pure conversation) to almost always
  (a pure uniform sampler).

### 2.1 Actors and operations

84 inputs (`src/input.rs`), grouped by who acts:

- **The person**, as the models hear them: microphone frames, VAD start/inference/end, STT
  start/interim/preflight/final/end, end-of-turn predictions, overlap verdicts, DTMF.
- **The developer, through the API**: `say`, `generate_reply`, `interrupt`, turn control
  (commit, clear, claim, release), floor holds and authorisation, fillers, runs; updating
  instructions, tools, history, models and options; handoffs and tasks; keyterms, speaker
  filter, IVR; attaching and toggling audio, video and transcription inputs and outputs; drain
  and close.
- **The developer's code, answering the framework**: hook results (on time and late), tool
  results and updates, turn commits, metrics.
- **The models and outputs**: LLM text, tool calls, end and failure; STT, EOT, overlap and TTS
  failures, holds and outages; realtime server events; playout started, progressed, finished;
  output resets; camera frames.
- **The room**: linked, unlinked, closed, state changes.

Every driver command variant has an input except `IvrCommand::AddTools`. The gaps are in the
arguments the inputs carry (section 5) and in the layer above the commands (section 6).

### 2.2 Phases of a conversation

When an input lands, the conversation is in one or more of these phases, read off the driver's
journal at that instant (`cells_by_input` in `src/drive/finish.rs`):

| Phase | Means |
|---|---|
| start | the agent is still initialising |
| listen | the agent listens and the person is quiet |
| person | the person is speaking |
| away | the person has been quiet long enough to be away |
| think | the agent is thinking (LLM or realtime generating) |
| speak | the agent is speaking |
| overlap | both speak at once: a barge-in or a backchannel being judged |
| tool | at least one tool call is running |
| turn hook | `on_user_turn_completed` is running in developer code |
| drain | a `drain()` is waiting for the current agent to settle |
| on_exit | an agent's `on_exit` is running (handoff, drain or close) |
| on_enter | an agent's `on_enter` is running (start or after a handoff) |
| closing | a close was requested |

The driver never reports `idle`, so it is not a column. Section 5 lists the phases the audit
found missing (the scheduler's among them) with the journal steps that would compute them.

### 2.3 What is drawn per seed

Everything below is drawn once per seed and holds for the run. Each kind of trouble has its own
coin and its own random stream, so some runs are clean and others full of one kind, and adding
a kind never moves another's draw (agent-store's swarm hunt).

| Who | Drawn per seed |
|---|---|
| Setup | about 30 switches, each on in roughly half the seeds (the sampler learns toward values that found new journal variants): STT, VAD, end-of-turn model, interruption detector, endpointing, turn mode, preemptive generation, backchannel threshold, transcription timeout, user-turn limits, speaker filter, keyterms, pipeline or realtime (with or without server turn detection), tools, max tool steps, hooks, TTS, pausable output, synchronised transcription, gap between speeches, IVR, AMD, room legs, video, connect latency, scheduler seed |
| The person | pace (40-250% of 120-210 words a minute), silence length (20-500%), how long they leave the agent to answer (20-500%), how long they sit in silence before speaking again (0.3-15 s), how often they pause first, hesitate, barge in (anywhere in a reply), backchannel (0-15% per beat while the agent speaks), make noise, have someone else speak, leave the call and come back |
| The developer | how often it calls the API between turns, per beat, and per beat while the agent speaks (three in ten programs are busy ones), how often the model it prompts calls tools, how often it scopes fillers |
| The call | average length, 30 s to 2 h flat over orders of magnitude, and how it ends |
| The providers | latency (25-400% of typical), late finals, mis-heard interims, split finals, lost end-of-turn predictions, how often the interruption detector is right; failures, each in 20-30% of seeds: LLM requests failing, a fatal STT failure (the driver does not reopen the recogniser), the overlap detector failing, the realtime socket dropping, the speaker resetting, the voice down for the whole run; waits aimed at the driver's armed deadlines |
| Randomness | uniform moves per conversation move, flat over seven decades from 1e-4 to 1e3 |

Every kind of trouble fired in at least 20 of the 1,024 seeds (an STT failure in 447, an LLM
failure in 319, an overlap-detector failure in 102, a lost end-of-turn prediction in 64, a
realtime drop or other server event in 198, a speaker reset in 501, a room event in 462, a TTS
stream failure in 20, the run-long voice outage in 508).

**Fixed in every run** (the audit, section 5, has the file references): the end-of-turn
detector is always streaming, never batch, and always the V1Mini model; turn detection is never
explicitly `vad` or `realtime_llm`; the AMD runs on its default thresholds; keyterm detection by
LLM is off; the false-interruption, backchannel-boundary, AEC-warmup, `min_duration` and
`discard_audio_if_uninterruptible` options keep their defaults; `close` is always
`user_initiated` without an error.

**Not modelled at all**: fallback adapters (`lk-agents-inference`); a realtime model paired with
a separate TTS; realtime tool calls; avatar and datastream outputs; a handoff that brings the
next agent's own models, options or history; a tool that hands off or calls the API itself
(fake tools only return values, errors, stops, or ask `wait_for_idle`); two people speaking to
the agent; text-only sessions.

**Only a quarter of seeds can speak.** The agent's audio plays only under a pipeline with TTS
(TTS on and no realtime model: about one seed in four), so every "speak" and "overlap" cell is
capped near a quarter of the seeds.

## 3. Promises and the rules that check them

23 rules (`src/rules/`), each one guarantee with examples that must pass and counterexamples
that must fail. Failures are counted per seed over 1,024 seeds, split by how random the run
was: **conversation** (under 0.01 uniform moves per conversation move, 295 seeds), **mixed**
(0.01 to 10, 437 seeds) and **uniform** (over 10, 292 seeds). A failure in a conversation seed
is one a real call produces; at the uniform end it may be a situation no real provider creates.
None has been triaged into driver bug vs rule imprecision yet (historically about one in four
was a driver bug).

| Promise | Rule | Guarantee | Failures: conversation / mixed / uniform |
|---|---|---|---|
| P1 | `api_calls_return` | every API call gets its result | 0 / 0 / 0 |
| P1 | `speech_settles` | every speech is done once nothing is left to answer | 13 / 26 / 10 |
| P1 | `drain_answers_once_settled` | `drain()` answers only after `on_exit` and every speech is done | 0 / 0 / 2 |
| P1 | `idle_respects_holds` | `wait_for_idle` waits for other callers' floor holds | 82 / 248 / 109 |
| P1 | `driver_exits_only_on_close` | the driver stops only after a close, never by panicking (incl. the no-progress watchdog) | 0 / 0 / 0 |
| P1 | harness: `session_settles` | the run goes quiet once everything is released | 25 / 172 / 177 |
| P2 | `interruption_has_cause` | every interruption matches an input that allows it | 39 / 307 / 383 |
| P2 | `interrupted_handle_is_journalled` | every interrupted handle has its `Interrupted` step | 32 / 5,406 / 6,588 |
| P2 | `playback_requires_authorization` | nothing speculative is played before it is authorised | 0 / 0 / 0 |
| P2 | `no_premature_completion` | a streamed `say` does not complete while its input is open | 28 / 139 / 192 |
| P2 | `agent_speaking_span_follows_state` | the speaking span brackets the speaking state | 0 / 0 / 0 |
| P3 | `audio_conservation` | played audio is an in-order prefix of synthesised audio | 0 / 0 / 0 |
| P3 | `text_conservation` | the transcript forwarded is a prefix of the speech's text, all of it on completion | 13 / 43 / 68 |
| P3 | `synthesized_speech_is_played` | a completed speech that synthesised text played audio | 1 / 3 / 1 |
| P3 | `forwarded_while_enabled` | every mic frame reaches every open STT/VAD while input is on, none while off | 101,537 / 78,304 / 1,243 (one per frame; see 9) |
| P3 | `dtmf_reports_what_was_pressed` | DTMF answers with the keys pressed | 0 / 0 / 0 |
| P4 | `speech_done_once` | one `SpeechDone` per speech, after its creation, matching its handle | 195 / 9,261 / 11,051 |
| P4 | `tool_call_reaches_runner` | a hosted tool call reaches the runner, an unhosted one does not | 2 / 8 / 9 |
| P4 | `tool_lifecycle` | each tool call starts once, ends once, settles before its speech is done | 0 / 2 / 7 |
| P4 | `on_exit_once_per_frame` | each agent's `on_exit` runs at most once | 0 / 0 / 0 |
| P4 | `output_replacement_is_clean` | a replaced output receives no further calls | 70 / 233 / 497 |
| P5 | `provider_failure_is_accounted` | every injected failure is retried, surfaced or closes the session | 10 / 34 / 31 |
| P5 | `speech_error_is_injected` | a failed speech traces back to an injected failure or an explicit refusal | 1 / 904 / 1,134 |
| P5 | `provider_streams_closed` | every LLM, TTS and STT stream is closed by the end | 1 / 0 / 0 |
| P5 | soak: residue | after a long call and release, nothing opened is still live | green on 8 long calls, but 5 of them hear the person only 1-7 times (section 9); an earlier run had speeches, tools and fillers piling up (peaks 35, 54, 70, 79 by quarter) |

Every rule is exercised. Promises with no rule are listed per subsystem in section 5.

## 4. Operation x phase

Seeds (of 1,024) in which each input landed in each phase. `#` 20 or more, `+` 1 to 19, `.`
never. Exact numbers are in the appendix.

**Reading it.** For the 40 developer API inputs (520 cells): 435 well covered, 77 rare, 8
never, all while the agent starts: `InterruptSpeech`, `SetAllowInterruptions`, `ReleaseFloor`,
`FillerStart`, `FillerStop`, `SetRunOutput`, `FeedAudio` (each needs something that does not
exist yet at start: a speech, a hold, a tool, a stream) and `Close`. For comparison, the same table
had 24 never with the earlier two separate modes, and 98 to 136 with conversations alone.
"Speak" and "overlap" stay `+` for most inputs because only a quarter of seeds can speak
(section 2.3).

**The person, as the models hear them**

| input | start | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `MicFrames` | # | # | # | # | # | # | # | # | # | # | # | # | # |
| `VadStart` | + | # | # | # | # | # | + | # | + | # | # | # | + |
| `VadInference` | + | # | # | # | # | # | # | # | # | # | # | # | # |
| `VadEnd` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `SttStart` | # | # | # | # | # | # | # | # | + | # | # | # | # |
| `SttInterim` | . | # | # | # | # | # | + | # | + | # | # | # | # |
| `SttPreflight` | . | # | # | # | # | + | + | # | + | # | + | # | # |
| `SttFinal` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `SttEnd` | . | # | # | # | # | # | + | # | + | # | # | # | # |
| `EotAnswer` | . | # | # | + | # | + | + | # | + | # | + | # | + |
| `OverlapVerdict` | + | # | # | # | # | + | + | # | . | # | + | # | # |
| `Dtmf` | # | # | # | # | # | # | # | # | + | # | # | # | # |

**Developer: speech and turns**

| input | start | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Say` | # | # | # | # | # | # | # | # | # | # | # | # | # |
| `GenerateReply` | # | # | # | # | # | # | # | # | # | # | # | # | # |
| `Interrupt` | # | # | # | # | # | # | # | # | + | # | # | # | # |
| `InterruptSpeech` | . | # | # | # | # | # | # | # | + | # | # | # | + |
| `SetAllowInterruptions` | . | # | # | # | # | # | # | # | + | # | # | # | + |
| `CommitUserTurn` | # | # | # | # | # | # | # | # | + | # | # | # | # |
| `ClearTurn` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `ClaimUserTurn` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `ReleaseUserTurn` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `WaitForIdle` | # | # | # | # | # | # | + | # | + | # | # | # | # |
| `HoldFloor` | # | # | # | # | # | # | # | # | # | # | # | # | # |
| `ReleaseFloor` | . | # | # | # | # | + | + | # | + | # | # | # | # |
| `PauseAuthorization` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `ResumeAuthorization` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `FillerStart` | . | # | # | # | # | # | # | # | + | # | # | # | # |
| `FillerStop` | . | + | # | # | # | + | + | # | + | # | + | # | + |
| `Run` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetRunOutput` | . | # | # | # | # | # | # | # | + | # | # | # | + |

**Developer: agent and options**

| input | start | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `UpdateInstructions` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `UpdateTools` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `UpdateChatCtx` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `UpdateModels` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `Configure` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `Handoff` | # | # | # | # | # | # | # | # | + | # | # | # | # |
| `Task` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetStaticKeyterms` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetSpeakerFilter` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `Ivr` | + | # | # | # | # | # | # | # | + | # | # | # | # |

**Developer: inputs and outputs**

| input | start | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `SetAudioInput` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetAudioOutput` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetVideoInput` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetTranscriptionOutput` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetIoEnabled` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `Recorder` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `PushVideo` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `FeedText` | + | # | # | # | # | # | + | # | + | # | # | # | # |
| `FeedAudio` | . | # | # | # | # | # | # | # | + | # | # | # | # |
| `FeedEnd` | + | # | # | # | # | # | # | # | + | # | # | # | # |

**Developer code answering the framework (hooks, tools)**

| input | start | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `HookDone` | . | # | # | # | # | + | + | # | # | # | # | # | # |
| `HookDoneLate` | . | # | # | # | # | # | # | # | + | # | # | # | # |
| `ToolAnswered` | . | # | # | # | # | # | + | # | + | # | + | # | # |
| `ToolUpdate` | . | # | # | # | # | + | + | # | + | # | + | # | # |
| `CommitTurn` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `HostMetrics` | + | # | # | # | # | # | # | # | + | # | # | # | # |

**Lifecycle**

| input | start | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Drain` | # | # | # | # | # | # | + | # | + | # | # | # | # |
| `Close` | . | # | # | # | # | + | + | # | + | # | + | # | # |

**Models, outputs, camera**

| input | start | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `LlmText` | . | # | # | # | # | # | + | # | + | # | + | # | + |
| `LlmToolCall` | . | # | # | # | # | # | # | # | # | # | # | # | + |
| `LlmEnd` | . | # | # | # | # | # | + | # | + | # | + | # | + |
| `LlmFail` | . | + | # | # | # | # | + | # | + | # | + | # | . |
| `SttFail` | + | # | # | # | # | + | + | # | + | # | # | # | + |
| `EotFail` | . | + | # | . | + | + | + | + | + | + | + | # | . |
| `OverlapFail` | + | # | # | # | # | + | + | # | + | # | + | # | # |
| `TtsHold` | . | + | + | + | + | + | + | + | . | + | . | + | . |
| `TtsRelease` | . | + | + | + | + | + | . | # | . | + | . | + | + |
| `TtsFail` | . | . | + | . | # | + | . | # | . | + | . | + | . |
| `TtsOutage` | # | # | # | # | # | # | # | # | + | # | # | # | # |
| `RealtimeEvent` | + | # | # | # | + | . | . | + | + | # | # | # | + |
| `PlayoutStarted` | . | # | # | + | # | + | + | # | + | # | + | # | . |
| `PlayoutProgressed` | . | + | # | + | # | # | # | # | . | # | + | # | + |
| `PlayoutFinished` | . | + | # | + | # | # | + | # | . | # | + | # | + |
| `SinkReset` | + | # | # | # | # | # | # | # | + | # | # | # | # |
| `VideoFrames` | + | # | # | # | # | # | # | # | # | # | # | # | # |

**Room**

| input | start | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Room` | + | # | # | # | # | + | + | # | + | # | # | # | # |

## 5. Subsystem audit

The driver, subsystem by subsystem: the states that matter to a person or a developer (and the
journal steps that would make each a phase column), the promises nobody checks, what the
fuzzer never varies, and what it never sends. Paths are under `rust/crates/`.

### 5.1 Speech scheduling (`lk-agents-voice/src/agent_activity/scheduling.rs`, `speech.rs`)

One speech on the floor at a time; the rest wait in `_speech_q` (every speech has the normal
priority, so it is a FIFO). A speech is `Queued`, `Preemptive`, `Current`, `Working` (step
spoken, tools running), `Parked` (its tool awaits an inline task) or `Discarded`. The loop is
idle, delaying out `min_consecutive_speech_delay`, held by authorization, or has a current
speech. Scheduling can be paused (a drain), refusing new speech except tool replies.

- **Phases that could be columns**, all from existing steps: *speech queued* (`SpeechCreated`
  until `Speech::Authorized` or `SpeechDone`), *queue backlog* (two or more queued), *speech
  gap* (`FloorArmed` until `FloorReleased`), *floor held by the developer*
  (`IdleAnswered{hold: Some}` until `FloorHoldReleased`), *authorization held*
  (`AuthorizationPaused` until `AuthorizationResumed`), *scheduling paused* (`SchedulingPaused`
  until `SchedulingPauseSettled`), *preemptive reply in flight* (`Emitted(PreemptiveGeneration)`
  until `Turn::PreemptiveReconciled` or `PreemptiveRefused`), *tool blocked on a task*
  (`Blocked{true}` until `Blocked{false}`).
- **Checked**: nothing speculative plays before it is authorised
  (`playback_requires_authorization`); every speech ends
  once (`speech_done_once`, `speech_settles`); `wait_for_idle` respects other holds
  (`idle_respects_holds`).
- **Not checked**:
  - at most one speech is authorised (on the floor) at any instant;
  - speeches are authorised in the order they were scheduled;
  - two consecutive playouts are at least `min_consecutive_speech_delay` apart;
  - nothing is authorised while authorization is held or another caller holds the floor;
  - a speech refused at scheduling has no visible effect: no state change, nothing played. The
    bug fixed on 2026-10-22 (a refused reply flipped the agent to `thinking` and livelocked
    the IVR) broke exactly this, and no rule caught it; the no-progress watchdog did.
  - a queued speech interrupted before it is authorised never plays;
  - a current speech that never plays out is ended by its watchdog (`WatchdogArmed`).
- **Never varied**: speech priority (the driver has none to vary); `min_consecutive_speech_delay`
  is varied.

### 5.2 Generation, synthesis and playout (`generation/`, `io.rs`, `transcription/synchronizer.rs`)

- **Phases**: *LLM request open* (`Generation::ChatRequested` until `ChatFinished`), *synthesis
  open* (`TtsRequested` until the segment ends), *retry armed* (`RetryArmed` until the attempt),
  *filler playing* (`FillerStarted` until `FillerStopped`), *paused playout* (see 5.4).
- **Checked**: played audio is a prefix of synthesised audio, transcript a prefix of the text,
  completed speeches played audio, streams closed (`audio_conservation`, `text_conservation`,
  `synthesized_speech_is_played`, `provider_streams_closed`).
- **Not checked**: the history item of an interrupted speech is truncated to what was played;
  an update (`update_instructions`, `update_tools`, `update_chat_ctx`, `UpdateModels`, options)
  applies to the next request and not to the one in flight; retries follow the configured
  count and backoff; a late result from a replaced stream never reaches the new one (only
  outputs are checked); filler stops when the reply starts.
- **Never varied**: `tts_text_transforms` (0% reached), `use_tts_aligned_transcript`, the realtime
  audio node.

### 5.3 Turn taking (`audio_recognition/`, `endpointing.rs`, `turn.rs`)

The turn machine is idle, speaking, ended, awaiting a detector, delaying, committed or manually
committed (`audio_recognition/mod.rs:648`). None of it is a column; "person" is only the
user state.

- **Phases**: *endpointing countdown* (`TimerArmed{EndpointingDelay|EotPredictionTimeout}` until
  fired or cancelled), *awaiting the detector* (`LegRequested{Eot}` until its verdict), *commit
  pending* (`Turn::CommitRequested` until `CommitAnswered` or `CommitSuperseded`), *user turn
  claimed* (`UserTurnClaims{depth > 0}`), *transcription timeout armed*, *user-turn limit armed*.
- **Not checked** (no rule at all): a finished turn gets a reply or a journalled non-reply
  verdict; no commit while the person speaks (manual commit aside); the commit comes between
  the minimum and maximum endpointing delay unless a detector is pending; `UserTurnExceeded`
  fires once per crossing; `TranscriptionTimeout` fires only after its timeout with no
  transcript.
- **Never varied**: preemptive generation off (`lk-agents-fuzz/src/drive/start.rs:29`, default on
  in `agent_activity/options.rs:213`); batch end-of-turn detectors (`start.rs:107`); detection
  `vad` and `realtime_llm` (`input.rs:411`); the detector model and per-language thresholds
  (`start.rs:136`); `endpointing.alpha`; `preemptive.max_speech_duration` and `max_retries`;
  `session_close_transcript_timeout`.

### 5.4 Interruptions, false interruptions, backchannels (`agent_activity/pause.rs`, `overlap.rs`)

- **Phases**: *speech paused* (`PlayoutGate{Pause}` until resume, `AgentFalseInterruption` or
  `Interrupted`), *resume timer armed* (`TimerArmed(FalseInterruption)`), *detector judging*
  (`LegRequested{Interruption}` until its verdict), *barge guard held*, *AEC warmup*.
- **Not checked**: no barge-in while interruptions are disabled, the speech disallows them, or
  under `min_words`/`min_duration` (`pause.rs:516-570`; `interruption_has_cause` only checks a
  user input preceded it); a paused speech resolves within its timeout; a backchannel inside
  the boundary is ignored; the input gate feeds silence only during uninterruptible speech or
  AEC warmup.
- **Never varied**: `resume_false_interruption`, `false_interruption_timeout`,
  `backchannel_boundary`, `discard_audio_if_uninterruptible`, `min_duration`, `aec_warmup`,
  interruption mode other than adaptive or unset.

### 5.5 Tools (`tool_executor.rs`, `agent_activity/tools.rs`)

- **Checked**: each call starts and ends once and settles before its speech is done; hosted
  calls reach the runner (`tool_lifecycle`, `tool_call_reaches_runner`).
- **Not checked**: `max_tool_steps` is respected; a tool cancelled by an interruption releases
  what it held; parallel calls with mixed outcomes each settle; a tool reply after a tool
  update is scheduled and finishes (`Tool::SessionToolsExtended` is never reached).
- **Never sent**: a tool that hands off (`fakes/tools.rs:76`, `agents` always empty, so
  `request_describe` and the `DescribeAgent` hook are unreachable); a tool that calls `say`,
  `generate_reply`, `update_*` or holds the floor (only `wait_for_idle` from a tool, so run
  ownership of a tool's hold, `Driver::run_own_hold`, is untested); realtime tool calls.

### 5.6 Handoffs, drains, tasks, hooks and runs (`handoff.rs`, `drain.rs`, `tasks.rs`, `hooks.rs`, `run_result.rs`)

- **Phases**: *hook running* per kind (`HookIssued{kind}` until `HookFinished{kind}`; four kinds
  beyond on_exit, on_enter and the turn hook have no column); *run open* (`Emitted(RunStarted)`
  until `RunSettled`). A *handoff in flight*, a *queued second handoff* and an *inline task
  active* cannot be computed: the handoff's begin, queueing and refusal
  (`begin_handoff`, `refuse_handoff`) and a task's entry and exit (`tasks.rs`) write no journal step.
- **Not checked**: a handoff is refused only when closing or its drain was refused; a
  successful handoff journals `AgentUpdated` and then `on_enter`; queued handoffs run in order;
  every hooked call with a bound finishes by its deadline (the soak's seed 2 Hook residue
  points here: a hosted `on_enter` has no deadline but an open run owns it,
  `Driver::issue_hook`); a task's awaiter is always released and its parent restored once; every
  `RunStarted` settles once, nested runs are refused, retries stay under the cap.
- **Never sent**: hook answers `Described` and `ToolsExecuted` (`apply.rs:141-148`), turn edits
  `ReplaceItems` and `UpdateUserMessage`; a completed, failed or cancelled `TaskHandle`
  (`apply.rs:490`), tasks with `preserve_function_call_history=false` or a parent; `capture_run`.

### 5.7 Close and errors (`agent_session/closing.rs`, `agent_session/mod.rs`)

- **Phases**: the close runs unwinding, exiting, draining, edge, committing, closing legs,
  closed (`Driver::close_advance`), with one step (`Close::Requested{drain}`) for all of it; a
  drain close and a hard close could be two columns from that step alone.
- **Not checked**: the session closes with `error` at exactly `max_unrecoverable_errors + 1`
  consecutive unrecoverable errors on one leg, and the count resets when the agent speaks or a
  transcript arrives (`ErrorCounts`); a close that is not an error waits for
  the trailing transcript; a second close joins the first.
- **Never sent**: close reasons other than `user_initiated`, a close carrying an error
  (`apply.rs:439`).

### 5.8 AMD, IVR, speaker filter, keyterms, user away

- **AMD**: defaults only (`start.rs:221-229`), no `Execute`/`Close` input; nothing checks that
  nothing plays while it holds authorization, that it emits exactly one result, or that
  authorization is always resumed. Its hold shares the `AuthorizationPaused` step with the
  developer's `PauseAuthorization`, so the two cannot be told apart in the journal.
- **IVR**: `AddTools` never sent; nothing checks that a loop or silence triggers exactly one
  reply and nothing triggers after close.
- **Speaker filter**: always detect and suppress (`start.rs:349`); no step for "learning"; no
  rule that background speakers never reach the turn or history.
- **Keyterms**: the LLM pass is off (`start.rs:161`), so `KeytermTimeout` and the interval are
  unreachable; no rule that static keyterms reach every new STT stream.
- **User away**: `Configure` never disables it; no rule that away is reported only after the
  timeout with both sides quiet.

### 5.9 The ten gaps that matter most

Ranked by what a person on the call or a developer would hit:

1. **No scheduling rule**: one speech on the floor at a time, in order, with the gap, never
   while authorization or the floor is held; a refused speech has no effect (5.1).
2. **No rule that a finished turn gets a reply or an explicit skip**, nor on endpointing delays
   (5.3).
3. **No rule that a barge-in respects `allow_interruptions`, `min_words`, `min_duration`**, and
   the false-interruption options are never varied (5.4).
4. **Updates apply to the right reply**: no rule for `update_*` against in-flight work, nor for
   the next agent's instructions and tools after a handoff (5.2, 5.6).
5. **The public API is never called** (section 6).
6. **Handoffs and tasks are not journalled**, so neither a column nor a rule can see them;
   tool-driven handoffs are unreachable (5.5, 5.6).
7. **Hooks without deadlines** may block a run forever (the soak's red seed) (5.6).
8. **Preemptive generation is never off**; batch EOT, explicit VAD and realtime-LLM detection
   are never set up (5.3).
9. **The error budget** (count, threshold, reset) has no rule (5.7).
10. **Realtime tool calls and a realtime model with a separate TTS** are not modelled (2.3).

## 6. Why the fuzzer goes around the public API

The fuzzer builds the driver with the internal `DriverBuilder` (`src/drive/start.rs:131`) and
sends each input as a driver `Command` (`src/drive/apply.rs:515`). The public `SessionApi`
(`lk-agents-voice/src/agent_session/api.rs`, `api/host.rs`) is what Python and Node call: it
parses arguments, validates them, builds the command and owns room I/O. None of that runs
under the fuzzer: 1,161 lines at 0%, plus `room_io/` at 0%.

There is no principled reason. `TEST_STRATEGY.md` says the opposite ("actual public API
payloads are the inputs to the system under test", "do not bypass the real interface"). The
methods are plain Rust (`host_new` takes trait objects and a `SessionRoom`; `lk-agents-rtc` has
a fake room), so nothing technical forces the shortcut. Two things make the switch more than a
rename:

- **Construction**: `host_new` takes the session's arguments as the host passes them; the
  harness's setups map onto them, and room events would then go through the real room I/O over
  the fake room instead of being injected.
- **Observation**: the rules need every journal step stamped in order with the fakes' records.
  The harness gets that by installing an observer on the driver's trace, which `SessionApi`
  exposes only as a `steps()` snapshot. Polling the snapshot loses the interleaving; reaching
  into the trace is a test-only API. This is the design decision to take before the switch.

What the switch would reach: argument validation (`ivr_start(max_silence=)`, `aclose(reason=)`,
`set_io_enabled` leg names, `update_turn_options` and its empty-patch no-op, `push_video`'s
`NotAnImage`), `set_aec_warmup`, `hook_described`, `hook_tools_executed`, set/unset participant,
the `amd_*` calls, and room I/O.

## 7. How the search was chosen

Measured on 2026-10-22/23, on reach only: journal variants, distinct edges (consecutive journal
steps), (call x phase) cells and code lines. Reach does not depend on the rules, so it can
compare searches before the rules are triaged. Bug-finding is the better yardstick, and
section 12 says how it will be measured.

| Search | Seeds | Wall | Variants | Edges | Developer cells never reached |
|---|--:|--:|--:|--:|--:|
| two separate modes: uniform sampler + conversation (before) | 256 + 256 | 14 s | 159 | 2,913 | 24 of 546 |
| conversation alone, with troubles | 512 | 21 s | 158 | 2,293 | 98 to 136 of 520 |
| one generator, randomness flat over a probability (0-100%) | 1,024 | 6 s | 156 | 3,627 | not measured |
| one generator, randomness flat over seven decades | 1,024 | 18 s | 159 | 3,752 | 11 of 520 |
| **the same, plus barge-ins anywhere, backchannels, patience, the call's end as its own dial (kept)** | **1,024** | **25 s** | **159** | **4,499** | **8 of 520** |

By decade of randomness, at 1,024 seeds:

| Uniform moves per conversation move | Seeds | Wall | Variants | Edges | Cells | Reached only here |
|---|--:|--:|--:|--:|--:|---|
| 1e-4 to 1e-3 | 143 | 4.6 s | 158 | 1,945 | 724 | |
| 1e-3 to 1e-2 | 152 | 5.0 s | 157 | 2,038 | 778 | |
| 1e-2 to 1e-1 | 146 | 3.3 s | 155 | 2,333 | 871 | |
| 1e-1 to 1 | 143 | 2.3 s | 156 | 2,868 | 914 | |
| 1 to 10 | 148 | 3.0 s | 154 | 2,932 | 932 | |
| 10 to 100 | 149 | 3.4 s | 151 | 2,793 | 935 | |
| 100 to 1,000 | 143 | 3.1 s | 153 | 2,918 | 925 | |

What this says:

- Neither end is enough. Before the person listened properly, only near-pure conversations
  reached barge-ins and backchannels; now every decade does, and the middle and the uniform end
  reach the most edges and (call x phase) cells. Every decade adds something, so no decade is
  dropped and none is weighted.
- Ending the call was the biggest waste: with random moves allowed to close it, half the runs
  ended early (median at input 386 of 1,500), almost all by a random `Close` or room event. With
  the call's end as its own dial and random moves never ending it, 324 of 1,024 end early: 135
  hang up on purpose, 167 are closed by the session's own error policy after model failures,
  22 by the person disconnecting.
- The person used to stop waiting at the first 100 ms the driver did nothing, which could hide
  a stuck agent behind a new turn. They now wait their own patience (0.3-15 s), whatever the
  driver does; catching a silence the agent had no right to leave is a rule's job (section 9).
- A flat draw of a probability fails: at 80% coherence a run already makes about 250 uniform
  moves and one of them closes the call, so nearly every run is short and barge-ins went
  unreached. Flat over orders of magnitude gives every regime the same share of seeds.
- What only uniform moves used to reach (drains, model failures mid-reply, room events, output
  resets, camera frames) is now also reached by the actors themselves, as troubles and the
  developer's calls.

Lessons taken from agent-store's simulation and gosim:

- **Draw the policy per seed** (gosim PR #28, "agent-store's hunt draws the policy per seed").
  Their partial-order scheduler found 2.6 times more bugs than uniform on average and 4 that
  uniform never found, but uniform still did best on 12 of 25 programs: keep both, draw per seed.
  Here: randomness per seed.
- **A coin per kind of trouble** (agent-store's swarm hunt), so adding a kind never moves
  another's draw.
- **Fail a livelock by name** (gosim's "no progress" watchdog): a run whose journal grows by
  100,000 steps without virtual time moving panics with that message, and the panic is a
  finding. Before, such a run hung the whole campaign.
- **Judge by known bugs, with the protocol fixed before the run** (gosim's benchmark,
  agent-store's held-out findings): not yet done here; section 12.
- **Bursts and same-instant orderings** (gosim's scheduling policy): not yet done here; section 8.

## 8. Biases left

Each is a choice the search makes for every run that a real call does not.

- **Every input settles before the next, and the scheduler is not POS.** The runtime is tokio's
  single-threaded one with a seeded RNG, which only changes tokio's own random choices, not the
  order tasks run in; and since each input settles before the next, the driver never sees two
  events in flight at once. Same-instant orderings and bursts without settling need the driver's
  input queue to accept several deliveries per instant; this is where gosim's partial-order
  sampling applies.
- **Time moves in 100 ms beats of 10 ms frames.** Frames never arrive one by one, and nothing
  lands inside a beat.
- **Fixed pools of values**: the person's sentences (`USER_TURNS`) and the agent's lines
  (`AGENT_TURNS`) are short fixed lists, and stock delays come from one list (`DELAYS_MS`).
- **Uniform over rows**: a uniform move weights every row the same, so an actor with more rows
  gets more moves (the developer has about 45 rows, the room one).
- **The setup is fixed where section 2.3 says so**: batch end-of-turn, explicit detection modes,
  AMD and interruption options, close reasons.
- **Run length**: every run has the same budget (1,500 inputs); short calls only happen when the
  call ends early. A per-seed budget would cover both.
- **One end for runs that outlast their budget**: those all end with the same release and close.
- **One person, one microphone**, and the person always starts the conversation.
- **The person is content-blind**: whether they barge in or backchannel does not depend on what
  the agent says, and what they say is one of nine sentences, never an answer to the agent.
- **The setup sampler learns** toward switches that found new ground. It is measured against
  `VOICE_UNIFORM_SETUPS=1`, which draws every setup uniformly.

## 9. How to read the numbers honestly

- A journal variant or a code line reached is not a guarantee checked. 159 of 160 journal
  variants are reached (missing: `Tool::SessionToolsExtended`),
  but section 5 lists dozens of promises nobody checks on them.
- Most seeds fail at least one rule, including conversation seeds. Until the failures are
  triaged, the driver's defect rate is unknown; that is the number a ship decision needs.
- The largest cluster is `forwarded_while_enabled`, one failure per microphone frame: audio
  stops reaching the recogniser. In five of the eight long soak calls the person is heard 1 to 7
  times in 12,000 inputs, and the soak still passes because it only checks for leaks. Either a
  driver bug or a fake's, and the first thing to triage.
- A stuck agent is not caught by anything yet. The person now waits their own patience, so a
  stuck agent shows as a long silence, but no rule says how long the agent may stay silent after
  a turn it should answer. That rule needs the per-operation bounds `TEST_STRATEGY.md` leaves
  undecided.
- No finite campaign shows an interaction is impossible. A `.` cell is a hole in the search; a
  `#` cell is a place the search goes, not proof the behaviour there is right.
- Detection rate is unmeasured: known bugs have never been held out to check the search finds
  them.

## 10. Code reached

Line coverage of `lk-agents-voice` over the whole suite (gate, replay corpus, soak, 1,024-seed
survey): **57.9% of 25,922 lines**.

| Area | Lines | Reached | Note |
|---|--:|--:|---|
| `audio_recognition/` | 1,708 | 94% | hearing: VAD, STT, end of turn, overlap |
| `trace.rs` | 151 | 92% | |
| `agent_session/` driver, cycle, mod | 1,670 | 92% | the loop |
| `generation/` | 1,990 | 90% | LLM and TTS requests, replans |
| `ivr/` | 248 | 90% | |
| `audio.rs` | 455 | 88% | legs and gates |
| `tool_executor.rs` | 557 | 87% | |
| `agent_activity/` | 7,200 | 86% | scheduling, interruption, tools, hooks, handoff |
| `transcription/synchronizer.rs` | 587 | 84% | |
| `io.rs` | 532 | 78% | the host output seam |
| `amd/` | 898 | 73% | |
| `endpointing.rs` | 266 | 57% | |
| `speech_handle.rs` | 552 | 41% | the handle the developer holds |
| `keyterm_detection.rs` | 464 | 21% | the LLM pass is never set up |
| **`agent_session/api.rs`, `api/host.rs`** | **1,161** | **0%** | **the public API (section 6)** |
| `room_io/` | 1,914 | 0% | the real room input and output (section 6) |
| `transcription/filters.rs`, `text_transforms.rs` | 543 | 0% | text transforms never configured |
| `background_audio.rs` | 547 | 0% | |
| `remote_session*` | 904 | 0% | |
| `workflows/` (builtin tasks) | 2,440 | 7% | host-side tasks; only the DTMF collector is used |
| `turn.rs`, `events.rs`, `host_events.rs` | 516 | 10% | |

The core loop is well reached and the search changes above barely moved this number: what is
left is not reachable by any search until the fuzzer goes through the public API and the room
I/O (section 6) and sets up what section 2.3 lists as fixed.

## 11. Outside this suite

Measured nowhere here, and each a separate source of production risk:

- The Python voice pipeline (`livekit-agents/livekit/agents/voice/`), which is what ships today:
  `docs/rust-core.md` says it does not call this driver.
- The binding from Python to the driver (`lk-agents-py`), plugins reached through host proxies.
- Threads, lanes and leases in agent-db, the A2A text door, and a text turn landing on a lane
  while a call is live (`storage/live.py`).
- Real provider behaviour: the fakes deliver exactly what the case says; nothing checks them
  against a real STT, LLM or TTS.
- `lk-agents-inference` (native gateway clients, fallback adapters), `lk-agents-media`,
  `lk-agents-rtc`.

## 12. Keeping it true

```bash
cd rust
# reach, failures and the operation x phase table, by decade of randomness (about 25 s)
VOICE_CELLS=1 VOICE_SEEDS=1024 cargo test --release -p lk-agents-fuzz --test survey -- --ignored --nocapture
# long calls, and what is still live after them
cargo test --release -p lk-agents-fuzz --test soak -- --ignored --nocapture
# code reached (a few minutes; needs cargo-llvm-cov and the llvm-tools component)
VOICE_SEEDS=1024 cargo llvm-cov --release -p lk-agents-fuzz --no-fail-fast --no-report -- --include-ignored
cargo llvm-cov report --release --summary-only | grep lk-agents-voice/src
```

Next:

- **A known-bug benchmark**, once the rules are triaged: each confirmed driver bug, put back one
  at a time, and how many seeds each search needs to catch it, with the protocol fixed before
  the first run. This replaces reach as the yardstick for search changes.
- **Generated**: add the phase columns of section 5, have each rule register the (promise,
  operation, phase) cells it judges, write sections 3 and 4 from what the survey observed, and
  fail a test when this file is stale (agent-store's `dst/COVERAGE.md` and `TestCoverageMatrix`).

## Appendix: operation x phase, exact seeds

Of 1,024 seeds. `Advance`, `AwaitAnswer`, `AwaitExit`, `AwaitSpeech` and `Settle` are the
harness waiting, not an actor's operation.

| input | start | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `Advance` | 1015 | 994 | 921 | 829 | 661 | 203 | 154 | 535 | 159 | 554 | 440 | 736 | 528 |
| `AwaitAnswer` | 2 | 206 | 446 | 206 | 299 | 34 | 23 | 294 | 3 | 315 | 36 | 291 | 10 |
| `AwaitExit` | 7 | 636 | 569 | 568 | 318 | 30 | 19 | 325 | 6 | 339 | 22 | 298 | 417 |
| `AwaitSpeech` | 0 | 139 | 382 | 154 | 311 | 29 | 21 | 294 | 2 | 299 | 21 | 275 | 11 |
| `ClaimUserTurn` | 10 | 455 | 630 | 352 | 386 | 51 | 28 | 345 | 9 | 336 | 45 | 389 | 313 |
| `ClearTurn` | 13 | 384 | 633 | 288 | 391 | 38 | 26 | 345 | 6 | 339 | 41 | 386 | 303 |
| `Close` | 0 | 459 | 210 | 406 | 52 | 7 | 2 | 130 | 1 | 79 | 6 | 25 | 229 |
| `CommitTurn` | 18 | 349 | 636 | 277 | 389 | 55 | 36 | 342 | 6 | 349 | 40 | 389 | 290 |
| `CommitUserTurn` | 49 | 493 | 717 | 380 | 422 | 58 | 37 | 375 | 6 | 347 | 60 | 463 | 365 |
| `Configure` | 8 | 370 | 635 | 274 | 388 | 47 | 29 | 343 | 4 | 341 | 38 | 378 | 307 |
| `Drain` | 47 | 366 | 575 | 300 | 342 | 30 | 19 | 312 | 8 | 370 | 39 | 362 | 363 |
| `Dtmf` | 44 | 461 | 684 | 343 | 403 | 52 | 32 | 346 | 9 | 339 | 50 | 419 | 369 |
| `EotAnswer` | 0 | 35 | 210 | 4 | 85 | 18 | 18 | 61 | 6 | 32 | 8 | 136 | 3 |
| `EotFail` | 0 | 7 | 64 | 0 | 19 | 2 | 2 | 13 | 1 | 5 | 2 | 33 | 0 |
| `FeedAudio` | 0 | 283 | 569 | 242 | 359 | 46 | 30 | 333 | 2 | 335 | 35 | 332 | 265 |
| `FeedEnd` | 1 | 378 | 685 | 303 | 384 | 38 | 22 | 354 | 8 | 376 | 50 | 363 | 431 |
| `FeedText` | 1 | 283 | 570 | 241 | 353 | 34 | 19 | 329 | 2 | 322 | 38 | 332 | 252 |
| `FillerStart` | 0 | 76 | 311 | 130 | 376 | 31 | 23 | 379 | 11 | 233 | 44 | 226 | 60 |
| `FillerStop` | 0 | 19 | 190 | 41 | 196 | 17 | 12 | 196 | 3 | 144 | 9 | 120 | 4 |
| `GenerateReply` | 103 | 652 | 738 | 477 | 402 | 45 | 29 | 356 | 33 | 378 | 162 | 507 | 370 |
| `Handoff` | 67 | 465 | 658 | 337 | 391 | 48 | 29 | 356 | 7 | 336 | 48 | 398 | 365 |
| `HoldFloor` | 90 | 628 | 742 | 491 | 515 | 126 | 97 | 444 | 47 | 475 | 200 | 522 | 386 |
| `HookDone` | 0 | 414 | 431 | 220 | 350 | 18 | 16 | 264 | 54 | 235 | 119 | 671 | 38 |
| `HookDoneLate` | 0 | 346 | 632 | 271 | 367 | 47 | 34 | 337 | 4 | 336 | 35 | 340 | 302 |
| `HostMetrics` | 11 | 368 | 637 | 289 | 391 | 48 | 30 | 345 | 4 | 338 | 40 | 374 | 306 |
| `Interrupt` | 48 | 465 | 662 | 318 | 392 | 56 | 41 | 358 | 5 | 376 | 48 | 407 | 373 |
| `InterruptSpeech` | 0 | 185 | 491 | 178 | 403 | 59 | 38 | 334 | 2 | 321 | 33 | 325 | 16 |
| `Ivr` | 11 | 390 | 655 | 289 | 383 | 62 | 34 | 337 | 7 | 338 | 40 | 362 | 314 |
| `LlmEnd` | 0 | 51 | 362 | 98 | 513 | 27 | 19 | 370 | 19 | 194 | 12 | 284 | 10 |
| `LlmFail` | 0 | 18 | 253 | 57 | 319 | 21 | 15 | 270 | 5 | 181 | 5 | 186 | 0 |
| `LlmText` | 0 | 25 | 326 | 72 | 459 | 23 | 14 | 317 | 14 | 173 | 16 | 235 | 2 |
| `LlmToolCall` | 0 | 53 | 444 | 204 | 552 | 52 | 38 | 451 | 22 | 286 | 73 | 353 | 7 |
| `MicFrames` | 611 | 843 | 896 | 602 | 584 | 131 | 96 | 471 | 87 | 456 | 201 | 703 | 328 |
| `OverlapFail` | 3 | 59 | 102 | 47 | 43 | 10 | 9 | 43 | 2 | 63 | 5 | 58 | 46 |
| `OverlapVerdict` | 2 | 57 | 108 | 47 | 39 | 9 | 9 | 40 | 0 | 60 | 7 | 56 | 53 |
| `PauseAuthorization` | 17 | 371 | 638 | 264 | 384 | 52 | 31 | 345 | 7 | 330 | 49 | 394 | 288 |
| `PlayoutFinished` | 0 | 10 | 55 | 12 | 53 | 35 | 19 | 57 | 0 | 37 | 2 | 24 | 2 |
| `PlayoutProgressed` | 0 | 7 | 50 | 10 | 34 | 53 | 22 | 44 | 0 | 28 | 2 | 26 | 1 |
| `PlayoutStarted` | 0 | 21 | 61 | 18 | 108 | 18 | 15 | 59 | 1 | 39 | 1 | 25 | 0 |
| `PushVideo` | 15 | 369 | 634 | 280 | 372 | 48 | 28 | 342 | 8 | 342 | 53 | 384 | 305 |
| `RealtimeEvent` | 7 | 198 | 191 | 126 | 3 | 0 | 0 | 1 | 14 | 90 | 21 | 152 | 17 |
| `Recorder` | 6 | 370 | 631 | 299 | 382 | 43 | 27 | 343 | 12 | 336 | 50 | 379 | 319 |
| `ReleaseFloor` | 0 | 451 | 411 | 364 | 251 | 16 | 9 | 244 | 3 | 173 | 26 | 257 | 147 |
| `ReleaseUserTurn` | 15 | 371 | 777 | 310 | 409 | 51 | 31 | 364 | 9 | 353 | 54 | 396 | 413 |
| `ResumeAuthorization` | 12 | 418 | 708 | 321 | 428 | 51 | 33 | 366 | 6 | 373 | 54 | 383 | 406 |
| `Room` | 14 | 214 | 462 | 166 | 242 | 15 | 9 | 250 | 4 | 253 | 26 | 251 | 285 |
| `Run` | 14 | 374 | 625 | 289 | 377 | 48 | 31 | 341 | 10 | 337 | 45 | 368 | 295 |
| `Say` | 102 | 635 | 745 | 508 | 462 | 61 | 41 | 399 | 27 | 416 | 155 | 513 | 378 |
| `SetAllowInterruptions` | 0 | 166 | 481 | 168 | 375 | 44 | 27 | 329 | 6 | 316 | 31 | 305 | 16 |
| `SetAudioInput` | 10 | 354 | 648 | 277 | 389 | 49 | 28 | 358 | 6 | 339 | 44 | 389 | 302 |
| `SetAudioOutput` | 12 | 354 | 634 | 264 | 388 | 62 | 42 | 341 | 6 | 335 | 51 | 373 | 309 |
| `SetIoEnabled` | 11 | 368 | 644 | 282 | 369 | 47 | 28 | 353 | 7 | 356 | 61 | 367 | 297 |
| `SetRunOutput` | 0 | 160 | 458 | 179 | 382 | 54 | 36 | 334 | 2 | 307 | 29 | 310 | 17 |
| `SetSpeakerFilter` | 15 | 366 | 625 | 284 | 366 | 45 | 24 | 331 | 5 | 329 | 44 | 363 | 314 |
| `SetStaticKeyterms` | 14 | 357 | 634 | 280 | 388 | 49 | 30 | 347 | 5 | 349 | 46 | 375 | 287 |
| `SetTranscriptionOutput` | 14 | 404 | 638 | 299 | 378 | 44 | 29 | 350 | 5 | 348 | 44 | 390 | 307 |
| `SetVideoInput` | 11 | 362 | 652 | 287 | 385 | 42 | 34 | 353 | 16 | 347 | 50 | 369 | 298 |
| `Settle` | 0 | 442 | 186 | 393 | 26 | 2 | 0 | 116 | 0 | 65 | 0 | 0 | 226 |
| `SinkReset` | 4 | 281 | 501 | 237 | 320 | 50 | 30 | 315 | 3 | 325 | 28 | 315 | 287 |
| `SttEnd` | 0 | 232 | 588 | 167 | 327 | 23 | 18 | 284 | 15 | 256 | 28 | 357 | 194 |
| `SttFail` | 7 | 156 | 447 | 145 | 262 | 9 | 9 | 265 | 3 | 255 | 23 | 297 | 13 |
| `SttFinal` | 1 | 217 | 532 | 152 | 296 | 23 | 20 | 263 | 13 | 240 | 25 | 342 | 133 |
| `SttInterim` | 0 | 155 | 532 | 130 | 274 | 25 | 19 | 247 | 18 | 236 | 20 | 316 | 134 |
| `SttPreflight` | 0 | 175 | 477 | 130 | 237 | 13 | 10 | 226 | 10 | 213 | 18 | 269 | 129 |
| `SttStart` | 37 | 436 | 593 | 326 | 411 | 53 | 33 | 337 | 16 | 341 | 58 | 461 | 26 |
| `Task` | 6 | 306 | 601 | 247 | 371 | 43 | 25 | 329 | 8 | 328 | 35 | 358 | 255 |
| `ToolAnswered` | 0 | 68 | 301 | 70 | 315 | 21 | 17 | 324 | 5 | 195 | 8 | 192 | 113 |
| `ToolUpdate` | 0 | 43 | 225 | 41 | 223 | 15 | 9 | 217 | 5 | 148 | 12 | 138 | 53 |
| `TtsFail` | 0 | 0 | 14 | 0 | 20 | 1 | 0 | 20 | 0 | 11 | 0 | 11 | 0 |
| `TtsHold` | 0 | 1 | 16 | 1 | 15 | 1 | 1 | 15 | 0 | 10 | 0 | 11 | 0 |
| `TtsOutage` | 79 | 278 | 508 | 244 | 308 | 35 | 30 | 301 | 5 | 327 | 36 | 306 | 290 |
| `TtsRelease` | 0 | 1 | 13 | 2 | 19 | 1 | 0 | 21 | 0 | 9 | 0 | 11 | 1 |
| `UpdateChatCtx` | 18 | 364 | 652 | 271 | 388 | 49 | 33 | 350 | 4 | 339 | 46 | 391 | 304 |
| `UpdateInstructions` | 12 | 371 | 631 | 295 | 388 | 52 | 29 | 335 | 7 | 338 | 43 | 383 | 309 |
| `UpdateModels` | 19 | 345 | 630 | 282 | 386 | 53 | 34 | 346 | 4 | 352 | 47 | 390 | 300 |
| `UpdateTools` | 9 | 364 | 645 | 280 | 399 | 53 | 37 | 355 | 9 | 335 | 36 | 381 | 301 |
| `VadEnd` | 4 | 177 | 506 | 129 | 276 | 42 | 35 | 222 | 15 | 204 | 36 | 360 | 26 |
| `VadInference` | 5 | 454 | 531 | 315 | 333 | 69 | 53 | 261 | 57 | 259 | 108 | 392 | 50 |
| `VadStart` | 4 | 380 | 356 | 258 | 284 | 44 | 16 | 217 | 18 | 185 | 34 | 375 | 8 |
| `VideoFrames` | 4 | 479 | 704 | 347 | 428 | 85 | 62 | 391 | 40 | 378 | 129 | 438 | 426 |
| `WaitForIdle` | 22 | 372 | 626 | 284 | 338 | 29 | 17 | 317 | 6 | 305 | 32 | 355 | 356 |
