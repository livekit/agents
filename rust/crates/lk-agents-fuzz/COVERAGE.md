# Voice driver coverage

What `lk-agents-fuzz` covers of the Rust voice driver (`lk-agents-voice`), and what it does not.
Every number is measured, not claimed: a 1,024-seed survey and `cargo llvm-cov` over the same
suite, at `56e265f06` (2026-10-25). Written by hand for now; section 12 says how to re-measure
and what makes it generated.

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

81 inputs (`src/input.rs`), grouped by who acts:

- **The person**, as the models hear them: microphone frames, VAD start/inference/end, STT
  start/interim/preflight/final/end, end-of-turn predictions, overlap verdicts, DTMF.
- **The developer, through the API**: `say`, `generate_reply`, `interrupt`, turn control
  (commit, clear, claim, release), floor holds and authorisation, fillers, runs; updating
  instructions, tools, history, models and options; handoffs and tasks; keyterms, IVR;
  attaching and toggling audio, video and transcription inputs and outputs; drain
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

The driver never reports `idle`, so it is not a column. Nor is `initializing` any more:
`Driver::open` moves the agent to listening inside `start()`, before any input can land, so the
agent's start shows in the on_enter column. Section 5 lists the phases the audit
found missing (the scheduler's among them) with the journal steps that would compute them.

### 2.3 What is drawn per seed

Everything below is drawn once per seed and holds for the run. Each kind of trouble has its own
coin and its own random stream, so some runs are clean and others full of one kind, and adding
a kind never moves another's draw (agent-store's swarm hunt).

| Who | Drawn per seed |
|---|---|
| Setup | about 30 switches, each on in roughly half the seeds (the sampler learns toward values that found new journal variants): STT, VAD, end-of-turn model, interruption detector, endpointing, turn mode, preemptive generation, backchannel threshold, transcription timeout, user-turn limits, keyterms, pipeline or realtime (with or without server turn detection), tools, max tool steps, hooks, TTS, pausable output, synchronised transcription, gap between speeches, IVR, AMD, room legs, video, connect latency, scheduler seed |
| The person | pace (40-250% of 120-210 words a minute), silence length (20-500%), how long they leave the agent to answer (20-500%), how long they sit in silence before speaking again (0.3-15 s), how often they pause first, hesitate, barge in (anywhere in a reply), backchannel (0-15% per beat while the agent speaks), make noise, have someone else speak, leave the call and come back |
| The developer | how often it calls the API between turns, per beat, and per beat while the agent speaks (three in ten programs are busy ones), how often the model it prompts calls tools, how often it scopes fillers |
| The call | average length, 30 s to 2 h flat over orders of magnitude, and how it ends |
| The providers | latency (25-400% of typical), late finals, mis-heard interims, split finals, lost end-of-turn predictions, how often the interruption detector is right; failures, each in 20-30% of seeds: LLM requests failing, a fatal STT failure (the driver does not reopen the recogniser), the overlap detector failing, the realtime socket dropping, the speaker resetting, the voice down for the whole run; waits aimed at the driver's armed deadlines |
| Randomness | uniform moves per conversation move, flat over seven decades from 1e-4 to 1e3 |

Every kind of trouble fired in at least 15 of the 1,024 seeds (an STT failure in 405, an LLM
failure in 297, an overlap-detector failure in 83, a lost end-of-turn prediction in 71, a
realtime drop or other server event in 155, a speaker reset in 470, a room event in 413, a TTS
stream failure in 15, the run-long voice outage in 468; each the input's busiest phase in the
appendix, so a lower bound).

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

28 rules (`src/rules/`), each one guarantee with examples that must pass and counterexamples
that must fail. Failures are counted per seed over 1,024 seeds, split by how random the run
was: **conversation** (under 0.01 uniform moves per conversation move, 295 seeds), **mixed**
(0.01 to 10, 437 seeds) and **uniform** (over 10, 292 seeds). A failure in a conversation seed
is one a real call produces; at the uniform end it may be a situation no real provider creates.
None has been triaged into driver bug vs rule imprecision yet (historically about one in four
was a driver bug).

| Promise | Rule | Guarantee | Failures: conversation / mixed / uniform |
|---|---|---|---|
| P1 | `api_calls_return` | every API call gets its result | 0 / 0 / 0 |
| P1 | `speech_settles` | every speech is done once nothing is left to answer | 17 / 16 / 2 |
| P1 | `drain_answers_once_settled` | `drain()` answers only after `on_exit` and every speech is done | 0 / 0 / 0 |
| P1 | `idle_respects_holds` | `wait_for_idle` waits for other callers' floor holds | 48 / 59 / 35 |
| P1 | `driver_exits_only_on_close` | the driver stops only after a close, never by panicking | 0 / 0 / 0 |
| P1 | `close_finishes` | a close, once begun, ends with the session's exit, never stalls with nothing left to wake it (the run waits for the exit however long it takes in virtual time; a virtual day without one is the finding) | 0 / 0 / 0 (new 2026-10-25; 1 of 5,120 seeds had a close past the former 30 s cut, now judged) |
| P1 | `virtual_time_advances` | the session never does unbounded work at one virtual instant (a livelock) | 0 / 0 / 0 |
| P1 | `every_task_finishes` | every task the session spawned finishes, without panicking, once it closed; a report names the spawning line | 0 / 0 / 0 |
| P1 | harness: `session_settles` | the run goes quiet once everything is released | 0 / 11 / 21 (2026-10-25; was 23 / 145 / 127 while a dead driver's refused releases counted as work) |
| P5 | `nothing_left_live` | once the driver exited, no hold, tool, hook call, filler, segment, utterance, prediction, realtime turn or say stream is still open | 15 / 77 / 60 (new 2026-10-25: holds in most, a detached tool in 14 of 1,024 seeds, a hook call in 2) |
| P2 | `interruption_has_cause` | every interruption matches an input that allows it | 22 / 87 / 140 |
| P2 | `interrupted_handle_is_journalled` | every interrupted handle has its `Interrupted` step | 5 / 45 / 46 |
| P2 | `playback_requires_authorization` | nothing speculative is played before it is authorised | 0 / 0 / 0 |
| P2 | `no_premature_completion` | a streamed `say` does not complete while its input is open | 15 / 70 / 69 |
| P2 | `agent_speaking_span_follows_state` | the speaking span brackets the speaking state | 0 / 0 / 0 |
| P3 | `audio_conservation` | played audio is an in-order prefix of synthesised audio | 0 / 0 / 0 |
| P3 | `text_conservation` | the transcript forwarded is a prefix of the speech's text, all of it on completion | 9 / 18 / 18 |
| P3 | `synthesized_speech_is_played` | a completed speech that synthesised text played audio | 3 / 3 / 2 |
| P3 | `forwarded_while_enabled` | every mic frame reaches every open STT/VAD while input is on, none while off | 91 / 135 / 91 |
| P3 | `dtmf_reports_what_was_pressed` | DTMF answers with the keys pressed | 0 / 0 / 0 |
| P4 | `speech_done_once` | one `SpeechDone` per speech, after its creation, matching its handle | 67 / 207 / 150 |
| P4 | `tool_call_reaches_runner` | a hosted tool call reaches the runner, an unhosted one does not | 0 / 2 / 1 |
| P4 | `tool_lifecycle` | each tool call starts once, ends once, settles before its speech is done | 0 / 7 / 4 |
| P4 | `on_exit_once_per_frame` | each agent's `on_exit` runs at most once | 0 / 0 / 0 |
| P4 | `output_replacement_is_clean` | a replaced output receives no further calls | 80 / 174 / 249 |
| P4 | `released_provider_unused` | a provider the core released is never named again | 0 / 0 / 0 |
| P5 | `provider_failure_is_accounted` | every injected failure is retried, surfaced or closes the session | 9 / 23 / 35 |
| P5 | `speech_error_is_injected` | a failed speech traces back to an injected failure or an explicit refusal | 1 / 6 / 19 |
| P5 | `provider_streams_closed` | every LLM, TTS and STT stream is closed by the end | 7 / 41 / 6 |
| P5 | soak: residue | after a long call and release, nothing opened is still live | red (2026-10-25): seed 6 leaves a floor hold. Every call must reach its close: one that stopped before it fails the soak |

Every rule is exercised. Promises with no rule are listed per subsystem in section 5.

## 4. Operation x phase

Seeds (of 1,024) in which each input landed in each phase. `#` 20 or more, `+` 1 to 19, `.`
never. Exact numbers are in the appendix.

**Reading it.** For the 40 developer API inputs (480 cells): 427 well covered, 53 rare, none
never. The start column is gone (section 2.2); with it the table had 8 never at `edc90fe24`, all
at start, 24 with the earlier two separate modes, and 98 to 136 with conversations alone.
Most rare cells are in the turn hook column. "Speak" and "overlap" counts stay in the tens of
seeds because only a quarter of seeds can speak (section 2.3).

**The person, as the models hear them**

| input | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `MicFrames` | # | # | # | # | # | # | # | # | # | # | # | # |
| `VadStart` | # | # | # | # | # | # | # | # | # | # | # | # |
| `VadInference` | # | # | # | # | # | # | # | # | # | # | # | # |
| `VadEnd` | # | # | # | # | # | # | # | # | # | # | # | # |
| `SttStart` | # | # | # | # | # | # | # | # | # | # | # | # |
| `SttInterim` | # | # | # | # | # | + | # | # | # | # | # | # |
| `SttPreflight` | # | # | # | # | # | + | # | + | # | + | # | # |
| `SttFinal` | # | # | # | # | # | + | # | # | # | # | # | # |
| `SttEnd` | # | # | # | # | # | + | # | # | # | # | # | # |
| `EotAnswer` | # | # | . | # | + | + | # | + | # | + | # | + |
| `OverlapVerdict` | # | # | # | # | + | + | # | . | # | + | # | # |
| `Dtmf` | # | # | # | # | # | # | # | + | # | # | # | # |

**Developer: speech and turns**

| input | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Say` | # | # | # | # | # | # | # | # | # | # | # | # |
| `GenerateReply` | # | # | # | # | # | # | # | # | # | # | # | # |
| `Interrupt` | # | # | # | # | # | # | # | + | # | # | # | # |
| `InterruptSpeech` | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetAllowInterruptions` | # | # | # | # | # | # | # | + | # | # | # | + |
| `CommitUserTurn` | # | # | # | # | # | # | # | + | # | # | # | # |
| `ClearTurn` | # | # | # | # | # | # | # | + | # | # | # | # |
| `ClaimUserTurn` | # | # | # | # | # | # | # | + | # | # | # | # |
| `ReleaseUserTurn` | # | # | # | # | # | # | # | + | # | # | # | # |
| `WaitForIdle` | # | # | # | # | # | + | # | + | # | # | # | # |
| `WaitForIdleAndHold` | # | # | # | # | # | # | # | # | # | # | # | # |
| `ReleaseIdleHold` | # | # | # | # | + | + | # | + | # | # | # | # |
| `PauseAuthorization` | # | # | # | # | # | # | # | + | # | # | # | # |
| `ResumeAuthorization` | # | # | # | # | # | # | # | + | # | # | # | # |
| `FillerStart` | # | # | # | # | # | + | # | + | # | # | # | # |
| `FillerStop` | + | # | # | # | + | + | # | + | # | + | # | + |
| `Run` | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetRunOutput` | # | # | # | # | # | # | # | + | # | # | # | + |

**Developer: agent and options**

| input | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `UpdateInstructions` | # | # | # | # | # | # | # | + | # | # | # | # |
| `UpdateTools` | # | # | # | # | # | # | # | + | # | # | # | # |
| `UpdateChatCtx` | # | # | # | # | # | # | # | + | # | # | # | # |
| `UpdateModels` | # | # | # | # | # | # | # | + | # | # | # | # |
| `Configure` | # | # | # | # | # | # | # | + | # | # | # | # |
| `Handoff` | # | # | # | # | # | # | # | + | # | # | # | # |
| `Task` | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetStaticKeyterms` | # | # | # | # | # | # | # | + | # | # | # | # |
| `Ivr` | # | # | # | # | # | # | # | + | # | # | # | # |

**Developer: inputs and outputs**

| input | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `SetAudioInput` | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetAudioOutput` | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetVideoInput` | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetTranscriptionOutput` | # | # | # | # | # | # | # | + | # | # | # | # |
| `SetIoEnabled` | # | # | # | # | # | # | # | + | # | # | # | # |
| `Recorder` | # | # | # | # | # | # | # | + | # | # | # | # |
| `PushVideo` | # | # | # | # | # | # | # | + | # | # | # | # |
| `FeedText` | # | # | # | # | # | + | # | + | # | # | # | # |
| `FeedAudio` | # | # | # | # | # | # | # | + | # | # | # | # |
| `FeedEnd` | # | # | # | # | # | # | # | + | # | # | # | # |

**Developer code answering the framework (hooks, tools)**

| input | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `HookDone` | # | # | # | # | + | + | # | # | # | # | # | # |
| `HookDoneLate` | # | # | # | # | # | # | # | + | # | # | # | # |
| `ToolAnswered` | # | # | # | # | + | + | # | + | # | + | # | # |
| `ToolUpdate` | # | # | # | # | + | + | # | + | # | + | # | # |
| `CommitTurn` | # | # | # | # | # | # | # | + | # | # | # | # |
| `HostMetrics` | # | # | # | # | # | # | # | + | # | # | # | # |

**Lifecycle**

| input | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Drain` | # | # | # | # | # | + | # | + | # | # | # | # |
| `Close` | # | # | # | # | + | + | # | + | # | + | # | # |

**Models, outputs, camera**

| input | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `LlmText` | # | # | # | # | + | + | # | + | # | + | # | + |
| `LlmToolCall` | # | # | # | # | # | # | # | # | # | # | # | # |
| `LlmEnd` | # | # | # | # | + | + | # | # | # | # | # | + |
| `LlmFail` | + | # | # | # | + | + | # | + | # | + | # | + |
| `SttFail` | # | # | # | # | # | + | # | + | # | + | # | # |
| `EotFail` | + | # | . | + | + | + | + | + | + | + | # | + |
| `OverlapFail` | # | # | # | # | + | + | # | + | # | + | # | # |
| `TtsHold` | + | + | + | + | + | + | + | . | + | . | + | . |
| `TtsRelease` | + | + | + | + | . | . | + | . | + | . | + | . |
| `TtsFail` | . | + | . | + | . | . | + | . | + | . | + | . |
| `TtsOutage` | # | # | # | # | # | + | # | + | # | # | # | # |
| `RealtimeEvent` | # | # | # | . | . | . | . | + | # | + | # | + |
| `PlayoutStarted` | # | # | + | # | + | + | # | . | # | + | # | . |
| `PlayoutProgressed` | + | # | + | + | # | # | # | . | # | + | # | . |
| `PlayoutFinished` | + | # | + | # | # | + | # | . | + | . | # | + |
| `SinkReset` | # | # | # | # | # | # | # | + | # | # | # | # |
| `VideoFrames` | # | # | # | # | # | # | # | # | # | # | # | # |

**Room**

| input | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Room` | # | # | # | # | + | + | # | + | # | # | # | # |

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
  (`IdleAnswered{hold: Some}` until `IdleHoldReleased`), *authorization held*
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
  every hooked call with a bound finishes by its deadline (a hosted `on_enter` has no deadline
  but an open run owns it, `Driver::issue_hook`); a task's awaiter is always released and its
  parent restored once; every `RunStarted` settles once, nested runs are refused, retries stay
  under the cap.
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

### 5.8 AMD, IVR, keyterms, user away

- **AMD**: defaults only (`start.rs:221-229`), no `Execute`/`Close` input; nothing checks that
  nothing plays while it holds authorization, that it emits exactly one result, or that
  authorization is always resumed. Its hold shares the `AuthorizationPaused` step with the
  developer's `PauseAuthorization`, so the two cannot be told apart in the journal.
- **IVR**: `AddTools` never sent; nothing checks that a loop or silence triggers exactly one
  reply and nothing triggers after close.
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
7. **Hooks without deadlines** may block a run forever (5.6).
8. **Preemptive generation is never off**; batch EOT, explicit VAD and realtime-LLM detection
   are never set up (5.3).
9. **The error budget** (count, threshold, reset) has no rule (5.7).
10. **Realtime tool calls and a realtime model with a separate TTS** are not modelled (2.3).

## 6. Why the fuzzer goes around the public API

The fuzzer builds the driver with the internal `DriverBuilder` (`src/drive/start.rs:131`) and
sends each input as a driver `Command` (`src/drive/apply.rs:515`). The public `SessionApi`
(`lk-agents-voice/src/agent_session/api.rs`, `api/host.rs`) is what Python and Node call: it
parses arguments, validates them, builds the command and owns room I/O. None of that runs
under the fuzzer: 1,296 lines at 0%, plus `room_io/` at 2% (only `link.rs`).

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

Re-measured at `56e265f06`, the kept search reaches 159 variants and 4,322 edges in 38 s, with
no developer cell never reached of 480 (the start column gone, section 2.2). The edges fell from
4,499 across the core's rewrite since `edc90fe24` (synchronous step, providers and outputs by
id, and the queue policy of section 8); the causes were not measured apart.

By decade of randomness, at 1,024 seeds (`56e265f06`):

| Uniform moves per conversation move | Seeds | Wall | Variants | Edges | Cells | Reached only here |
|---|--:|--:|--:|--:|--:|---|
| 1e-4 to 1e-3 | 143 | 9.9 s | 158 | 1,945 | 703 | |
| 1e-3 to 1e-2 | 152 | 9.0 s | 155 | 2,075 | 779 | |
| 1e-2 to 1e-1 | 146 | 6.4 s | 155 | 2,241 | 830 | |
| 1e-1 to 1 | 143 | 2.5 s | 156 | 2,643 | 873 | `Emitted::BackchannelConfirmed` |
| 1 to 10 | 148 | 2.2 s | 153 | 2,609 | 869 | |
| 10 to 100 | 149 | 3.3 s | 150 | 2,459 | 882 | |
| 100 to 1,000 | 143 | 4.4 s | 153 | 2,748 | 899 | |

By queue policy (section 8), drawn per seed: `Arrival` 508 seeds, 159 variants, 3,558 edges, 15
findings; `Pos` 516 seeds, 158 variants, 3,794 edges, 16 findings.

What this says:

- Neither end is enough. Before the person listened properly, only near-pure conversations
  reached barge-ins and backchannels; now every decade does, and the middle and the uniform end
  reach the most edges and (call x phase) cells. Every decade adds something, so no decade is
  dropped and none is weighted.
- Ending the call was the biggest waste: with random moves allowed to close it, half the runs
  ended early (median at input 386 of 1,500), almost all by a random `Close` or room event. With
  the call's end as its own dial and random moves never ending it, 324 of 1,024 end early: 135
  hang up on purpose, 167 are closed by the session's own error policy after model failures,
  22 by the person disconnecting (2026-10-23; the survey does not print this split).
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
- **Bursts and same-instant orderings** (gosim's scheduling policy): the case owns the driver's
  queue and half the seeds pick the next event by partial-order sampling (`drive::queue::Policy`);
  section 8.

## 8. Biases left

Each is a choice the search makes for every run that a real call does not.

- **The order is the queue's, not the runtime's.** Applying an input does not yield, so a case
  can build a burst the driver untangles in one cycle, and the case picks what the driver's queue
  delivers next: arrival order, or basic partial-order sampling over the event sources and the
  driver's task pool (`drive::queue::Policy`, half the seeds each). Each source keeps its own
  order, and full POS is not used: every event reaches one core, so it would degenerate to a
  random walk. Tasks still run in tokio's fixed order on a single thread.
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
- Most seeds fail at least one rule (884 of 1,024), including conversation seeds (202 of 295).
  Until the failures are triaged, the driver's defect rate is unknown; that is the number a ship
  decision needs.
- The widest clusters are `output_replacement_is_clean` (503 seeds), `speech_done_once` (424)
  and `forwarded_while_enabled` (317): audio stops reaching the recogniser. In four of the eight
  soak calls the person is heard 0 to 7 times (three of them over 12,000 inputs), which the soak
  does not judge: it only checks for leaks. Either a driver bug or a fake's, and the first thing
  to triage.
- Registered providers are not checked to be released. After the driver exits, the session's
  provider table (`lk_agents_abi::resources`) still holds entries in 285 of 300 runs: the data
  plane's sinks and the microphone in most (the runtime never releases them on exit), and in 116
  a model or output the host registered for a command the driver never took. The table dies with
  the session, so this is memory held until then, not a process leak. Checking it needs a
  read-only accessor on `Resources` and a decision on who releases the data plane's entries.
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
survey): **56.9% of 25,558 lines**.

| Area | Lines | Reached | Note |
|---|--:|--:|---|
| `audio_recognition/` | 1,700 | 94% | hearing: VAD, STT, end of turn, overlap |
| `trace.rs` | 152 | 92% | |
| `agent_session/` but the API and `host_events.rs` | 2,370 | 88% | the loop: driver, effects, signals, close |
| `generation/` | 2,000 | 89% | LLM and TTS requests, replans |
| `ivr/` | 247 | 89% | |
| `audio.rs` | 460 | 88% | legs and gates |
| `tool_executor.rs` | 575 | 87% | |
| `agent_activity/` | 5,990 | 86% | scheduling, interruption, tools, hooks, handoff |
| `transcription/synchronizer.rs` | 593 | 86% | |
| `io.rs` | 295 | 79% | the host output seam (the IO traits moved to `lk-agents-abi`) |
| `amd/` | 898 | 73% | |
| `endpointing.rs` | 266 | 58% | |
| `speech_handle.rs` | 573 | 40% | the handle the developer holds |
| `keyterm_detection.rs` | 566 | 24% | the LLM pass is never set up |
| **`agent_session/api.rs`, `api/host.rs`** | **1,296** | **0%** | **the public API (section 6)** |
| `room_io/` | 1,969 | 2% | the real room input and output (section 6); only `link.rs` |
| `transcription/filters.rs`, `text_transforms.rs` | 543 | 0% | text transforms never configured |
| `background_audio.rs` | 547 | 0% | |
| `remote_session*` | 946 | 0% | |
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

**The launch gate** is one command, `rust/crates/lk-agents-fuzz/launch-gate.sh`: the workspace
build, the voice core's unit tests, the fuzz suites (every contract's examples, counterexamples
and probe, the replay corpus, determinism, the regression tests), the 1,024-seed survey, a
4,096-seed campaign on the next window (`VOICE_GATE_SEEDS`), and the soak. It fails on any
finding of a contract not in `check::KNOWN_FINDINGS` (the open findings, listed so a new kind of
defect cannot hide among them), on a soak residue, and on a campaign that did not run. A new
finding's case is saved to `replay/` pinned to it, because a campaign seed only reproduces
inside its whole window. About three minutes on a laptop.

What it can and cannot miss:

- **Verdicts never depend on real time.** A livelock is counted in steps at one virtual instant
  (`NO_PROGRESS_STEPS`, twenty times the busiest instant measured), in the journal and in the
  run's own loop, and reported as `virtual_time_advances`. Every wait the harness makes is
  bounded in virtual time (`BUDGET_MS`), except the wait for a begun close, which runs until the
  session exits (a virtual day without an exit is `close_finishes`), and every loop in rounds
  (release, join), so a slow machine cannot change a verdict. Every run that reaches its ending
  is judged closed or reported.
- **A saved case replays its run.** A campaign saves the whole run, ending included; a case
  holding `Settle` (which only the runner sends) is replayed without being finished again.
- **A real hang ends the process, loudly.** A product task that never yields blocks the run's
  only thread, where no step count can see it; a watchdog thread beside it (`VOICE_HANG_S`, 600 s)
  aborts with the seed. It is a crash, not a verdict: rerun the seed with `VOICE_TRACE=1`.
- **Leaks**: tasks (`every_task_finishes`, with the spawning line), speeches (`speech_settles`),
  answers (`api_calls_return`), provider streams (`provider_streams_closed`), turn-owned work
  (`nothing_left_live`), and growth over forty turns (the soak). Not checked: the provider table
  (`lk_agents_abi::resources`), section 9.

```bash
cd rust
# reach, failures and the operation x phase table, by decade of randomness (about 40 s)
VOICE_CELLS=1 VOICE_SEEDS=1024 cargo test --release -p lk-agents-fuzz --test survey -- --ignored --nocapture
# long calls, and what is still live after them
cargo test --release -p lk-agents-fuzz --test soak -- --ignored --nocapture
# code reached (a few minutes; needs cargo-llvm-cov and the llvm-tools component; exits non-zero
# while the soak is red, the profiles are still written)
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

| input | listen | person | away | think | speak | overlap | tool | turn hook | drain | on_exit | on_enter | closing |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `Advance` | 1024 | 941 | 839 | 713 | 206 | 143 | 563 | 214 | 525 | 483 | 880 | 661 |
| `AwaitAnswer` | 136 | 369 | 146 | 258 | 25 | 21 | 240 | 7 | 194 | 25 | 287 | 9 |
| `AwaitExit` | 590 | 533 | 499 | 300 | 19 | 10 | 319 | 7 | 290 | 32 | 335 | 535 |
| `AwaitSpeech` | 79 | 328 | 105 | 260 | 20 | 12 | 246 | 6 | 214 | 18 | 260 | 9 |
| `ClaimUserTurn` | 400 | 547 | 311 | 346 | 41 | 27 | 307 | 8 | 259 | 34 | 404 | 364 |
| `ClearTurn` | 330 | 584 | 217 | 364 | 43 | 32 | 319 | 12 | 271 | 41 | 407 | 366 |
| `Close` | 466 | 403 | 390 | 165 | 10 | 7 | 235 | 1 | 173 | 7 | 156 | 401 |
| `CommitTurn` | 322 | 613 | 221 | 366 | 46 | 34 | 314 | 4 | 259 | 35 | 416 | 358 |
| `CommitUserTurn` | 487 | 691 | 331 | 425 | 57 | 37 | 355 | 12 | 284 | 54 | 561 | 457 |
| `Configure` | 315 | 594 | 219 | 358 | 41 | 27 | 323 | 11 | 278 | 42 | 411 | 366 |
| `Drain` | 338 | 532 | 244 | 325 | 25 | 17 | 307 | 6 | 288 | 32 | 402 | 445 |
| `Dtmf` | 445 | 654 | 304 | 389 | 43 | 26 | 320 | 8 | 276 | 47 | 484 | 446 |
| `EotAnswer` | 44 | 204 | 0 | 85 | 19 | 19 | 50 | 14 | 30 | 15 | 160 | 3 |
| `EotFail` | 6 | 71 | 0 | 19 | 4 | 4 | 17 | 1 | 6 | 2 | 23 | 1 |
| `FeedAudio` | 209 | 522 | 182 | 337 | 29 | 22 | 310 | 9 | 272 | 37 | 350 | 285 |
| `FeedEnd` | 310 | 678 | 221 | 361 | 38 | 25 | 325 | 7 | 310 | 42 | 375 | 560 |
| `FeedText` | 209 | 511 | 178 | 309 | 25 | 17 | 284 | 7 | 245 | 26 | 338 | 283 |
| `FillerStart` | 73 | 266 | 109 | 328 | 21 | 12 | 335 | 7 | 181 | 39 | 224 | 66 |
| `FillerStop` | 13 | 145 | 25 | 162 | 8 | 7 | 152 | 1 | 102 | 2 | 111 | 4 |
| `GenerateReply` | 630 | 699 | 448 | 383 | 52 | 32 | 324 | 28 | 308 | 142 | 608 | 465 |
| `Handoff` | 440 | 660 | 302 | 388 | 44 | 27 | 331 | 9 | 276 | 45 | 487 | 457 |
| `WaitForIdleAndHold` | 589 | 740 | 447 | 533 | 117 | 87 | 458 | 41 | 428 | 208 | 615 | 484 |
| `HookDone` | 428 | 456 | 214 | 406 | 13 | 6 | 272 | 79 | 214 | 123 | 768 | 60 |
| `HookDoneLate` | 276 | 569 | 194 | 344 | 34 | 24 | 305 | 6 | 277 | 34 | 335 | 365 |
| `HostMetrics` | 315 | 600 | 227 | 360 | 40 | 29 | 323 | 11 | 258 | 40 | 410 | 377 |
| `Interrupt` | 445 | 640 | 295 | 393 | 48 | 32 | 328 | 14 | 299 | 43 | 478 | 460 |
| `InterruptSpeech` | 133 | 427 | 147 | 376 | 37 | 24 | 315 | 3 | 252 | 26 | 322 | 23 |
| `Ivr` | 345 | 602 | 227 | 372 | 41 | 27 | 326 | 8 | 260 | 49 | 417 | 393 |
| `LlmEnd` | 67 | 340 | 72 | 519 | 15 | 8 | 340 | 34 | 154 | 27 | 358 | 16 |
| `LlmFail` | 9 | 220 | 47 | 297 | 11 | 8 | 251 | 4 | 131 | 9 | 197 | 2 |
| `LlmText` | 37 | 289 | 63 | 464 | 12 | 8 | 292 | 19 | 137 | 11 | 287 | 4 |
| `LlmToolCall` | 73 | 447 | 200 | 568 | 36 | 27 | 462 | 29 | 261 | 85 | 451 | 20 |
| `MicFrames` | 844 | 908 | 557 | 624 | 116 | 83 | 473 | 120 | 381 | 218 | 834 | 311 |
| `OverlapFail` | 35 | 83 | 28 | 38 | 2 | 2 | 39 | 2 | 40 | 4 | 63 | 62 |
| `OverlapVerdict` | 39 | 88 | 26 | 42 | 10 | 9 | 41 | 0 | 43 | 6 | 52 | 56 |
| `PauseAuthorization` | 298 | 598 | 205 | 367 | 46 | 32 | 330 | 7 | 261 | 37 | 413 | 365 |
| `PlayoutFinished` | 10 | 35 | 13 | 30 | 35 | 18 | 26 | 0 | 19 | 0 | 20 | 2 |
| `PlayoutProgressed` | 7 | 47 | 11 | 18 | 52 | 25 | 28 | 0 | 25 | 1 | 22 | 0 |
| `PlayoutStarted` | 24 | 61 | 14 | 107 | 16 | 14 | 42 | 0 | 20 | 1 | 33 | 0 |
| `PushVideo` | 305 | 589 | 220 | 374 | 45 | 27 | 328 | 15 | 269 | 41 | 410 | 357 |
| `RealtimeEvent` | 155 | 128 | 91 | 0 | 0 | 0 | 0 | 12 | 60 | 16 | 139 | 12 |
| `Recorder` | 326 | 598 | 224 | 360 | 45 | 25 | 309 | 12 | 269 | 34 | 395 | 371 |
| `ReleaseIdleHold` | 331 | 326 | 267 | 216 | 8 | 3 | 203 | 1 | 126 | 31 | 242 | 175 |
| `ReleaseUserTurn` | 309 | 737 | 223 | 385 | 44 | 25 | 327 | 8 | 283 | 50 | 429 | 504 |
| `ResumeAuthorization` | 355 | 678 | 243 | 400 | 51 | 32 | 345 | 15 | 298 | 52 | 422 | 500 |
| `Room` | 178 | 413 | 120 | 215 | 16 | 11 | 235 | 3 | 184 | 20 | 248 | 313 |
| `Run` | 303 | 597 | 211 | 369 | 45 | 27 | 331 | 12 | 276 | 43 | 409 | 360 |
| `Say` | 599 | 733 | 461 | 497 | 55 | 37 | 398 | 23 | 366 | 149 | 634 | 457 |
| `SetAllowInterruptions` | 130 | 437 | 137 | 352 | 38 | 26 | 289 | 4 | 246 | 30 | 325 | 18 |
| `SetAudioInput` | 295 | 601 | 212 | 358 | 50 | 26 | 314 | 8 | 269 | 32 | 396 | 369 |
| `SetAudioOutput` | 304 | 591 | 218 | 366 | 53 | 31 | 331 | 12 | 265 | 46 | 398 | 366 |
| `SetIoEnabled` | 310 | 599 | 218 | 354 | 43 | 30 | 307 | 5 | 275 | 33 | 394 | 372 |
| `SetRunOutput` | 115 | 438 | 131 | 363 | 43 | 27 | 291 | 1 | 231 | 20 | 310 | 19 |
| `SetStaticKeyterms` | 309 | 612 | 220 | 354 | 43 | 30 | 297 | 5 | 268 | 39 | 400 | 363 |
| `SetTranscriptionOutput` | 323 | 593 | 204 | 348 | 58 | 35 | 310 | 4 | 276 | 39 | 387 | 358 |
| `SetVideoInput` | 315 | 591 | 226 | 362 | 45 | 31 | 313 | 7 | 291 | 33 | 409 | 352 |
| `Settle` | 415 | 288 | 346 | 30 | 2 | 0 | 130 | 0 | 82 | 0 | 1 | 385 |
| `SinkReset` | 220 | 470 | 181 | 280 | 49 | 22 | 276 | 5 | 250 | 27 | 326 | 349 |
| `SttEnd` | 261 | 608 | 139 | 321 | 24 | 19 | 269 | 23 | 221 | 28 | 441 | 288 |
| `SttFail` | 150 | 405 | 108 | 236 | 20 | 13 | 242 | 6 | 184 | 17 | 286 | 167 |
| `SttFinal` | 214 | 550 | 114 | 296 | 24 | 19 | 248 | 27 | 177 | 24 | 413 | 140 |
| `SttInterim` | 144 | 550 | 111 | 288 | 20 | 15 | 226 | 22 | 191 | 25 | 390 | 149 |
| `SttPreflight` | 157 | 458 | 114 | 238 | 22 | 13 | 209 | 8 | 167 | 16 | 309 | 149 |
| `SttStart` | 538 | 621 | 382 | 445 | 54 | 34 | 343 | 28 | 292 | 59 | 620 | 153 |
| `Task` | 279 | 544 | 216 | 361 | 38 | 25 | 309 | 6 | 246 | 32 | 355 | 304 |
| `ToolAnswered` | 63 | 266 | 53 | 263 | 9 | 8 | 282 | 2 | 143 | 6 | 172 | 136 |
| `ToolUpdate` | 42 | 181 | 41 | 185 | 12 | 8 | 180 | 1 | 102 | 5 | 129 | 71 |
| `TtsFail` | 0 | 11 | 0 | 13 | 0 | 0 | 15 | 0 | 6 | 0 | 7 | 0 |
| `TtsHold` | 3 | 14 | 3 | 12 | 1 | 1 | 15 | 0 | 9 | 0 | 13 | 0 |
| `TtsOutage` | 285 | 468 | 202 | 292 | 27 | 19 | 282 | 7 | 268 | 21 | 400 | 352 |
| `TtsRelease` | 3 | 7 | 4 | 9 | 0 | 0 | 12 | 0 | 8 | 0 | 10 | 0 |
| `UpdateChatCtx` | 326 | 588 | 226 | 357 | 47 | 27 | 322 | 16 | 274 | 46 | 406 | 365 |
| `UpdateInstructions` | 315 | 588 | 224 | 377 | 47 | 33 | 314 | 7 | 264 | 38 | 406 | 365 |
| `UpdateModels` | 315 | 592 | 223 | 357 | 40 | 28 | 326 | 12 | 274 | 33 | 416 | 361 |
| `UpdateTools` | 316 | 595 | 219 | 367 | 46 | 26 | 324 | 13 | 273 | 36 | 400 | 362 |
| `VadEnd` | 176 | 549 | 114 | 295 | 41 | 40 | 221 | 29 | 181 | 43 | 422 | 205 |
| `VadInference` | 470 | 567 | 301 | 341 | 66 | 57 | 253 | 75 | 228 | 120 | 471 | 167 |
| `VadStart` | 405 | 362 | 254 | 277 | 51 | 22 | 214 | 26 | 159 | 31 | 440 | 147 |
| `VideoFrames` | 433 | 667 | 290 | 418 | 81 | 62 | 353 | 37 | 298 | 135 | 474 | 501 |
| `WaitForIdle` | 341 | 561 | 237 | 303 | 22 | 13 | 277 | 4 | 228 | 31 | 375 | 427 |
