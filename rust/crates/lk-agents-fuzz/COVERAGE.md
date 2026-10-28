# Voice driver coverage

What `lk-agents-fuzz` covers of the Rust voice driver (`lk-agents-voice`), and what it does not.
Every number is measured, not claimed: a 1,024-seed survey and `cargo llvm-cov` over the same
suite. Sections 6 and 10 describe the harness through the public API (2026-10-26); sections 7
to 9 were re-measured at `049e7e332` (2026-10-25), after a seed became the whole of its run; sections 2 to 5 and the appendix are
still the measurement at `56e265f06`. Written by hand for now; section 12 says how to re-measure
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
arguments the inputs carry (section 5); every input that is a host call is made as one (section
6).

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
| Setup | about 30 switches, each a coin (a die for three-way ones) drawn from the seed alone: STT, VAD, end-of-turn model, interruption detector, endpointing, turn mode, preemptive generation, backchannel threshold, transcription timeout, user-turn limits, keyterms, pipeline or realtime (with or without server turn detection), tools, max tool steps, hooks, TTS, pausable output, synchronised transcription, gap between speeches, IVR, AMD, room legs, video, connect latency, scheduler seed |
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
explicitly `vad` or `realtime_llm`; the AMD runs on its default thresholds; the false-interruption, backchannel-boundary, AEC-warmup, `min_duration` and
`discard_audio_if_uninterruptible` options keep their defaults; `close` is always
`user_initiated` without an error.

**Not modelled at all**: fallback adapters (`lk-agents-inference`); a realtime model paired with
a separate TTS; realtime tool calls; avatar and datastream outputs; a handoff that brings the
next agent's own models, options or history; a tool that hands off or calls the API itself
(fake tools only return values, errors, stops, or ask `wait_for_idle`); two people speaking to
the agent; text-only sessions.

**About half the seeds are heard.** The person heard the agent (a segment started playing, on any
output) in 17,759 of 32,768 runs (the survey's "runs where the person heard the agent", at
2026-10-27). The rest are mostly runs a uniform sampler closes early, realtime runs that end
early, the one AMD call in eight that rings out, and voice outages; a pipeline almost always has
a recogniser and a voice (nine in ten each), outages end, and a leg the program turns off comes
back.

## 3. Promises and the rules that check them

37 rules (`src/rules/`), each one guarantee with examples that must pass and counterexamples
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
| P5 | `nothing_left_live` | once the driver exited, no hold, tool, hook call, filler, segment, utterance, prediction, realtime turn or say stream is still open | 0 / 0 / 0 (2026-10-25; was 15 / 77 / 60: holds, hook calls, then a detached tool the driver could not cancel) |
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
| P4 | `output_replacement_is_clean` | a replaced output receives no further calls but a running step's and a synchroniser's closing flushes | 80 / 174 / 249 |
| P4 | `released_provider_unused` | a provider the core released is never named again | 0 / 0 / 0 |
| P4 | `nodes_are_the_active_agents` | every node the session opens (`llm`, `tts`, `stt`, `transcription`, `realtime_audio_output`) is an override of the agent active then | 0 / 0 / 0 (new 2026-10-27) |
| P5 | `provider_failure_is_accounted` | every injected failure is retried, surfaced (on the speech or as its error event), abandoned with an interrupted speech, or closes the session | 9 / 23 / 35 |
| P5 | `speech_error_is_injected` | a failed speech traces back to an injected failure or an explicit refusal | 1 / 6 / 19 |
| P5 | `provider_streams_closed` | every LLM, TTS and STT stream is closed by the end | 7 / 41 / 6 |
| P1 | `speech_is_heard` | a speech pushes audio within 15 s of its creation and of the scheduler letting it go, ends, or waits on something that still holds; one that ended unheard was cut by the person, the host, the server or the close (section 13) | 0 / 0 / 0 (new 2026-10-26) |
| P1 | `pause_resolves` | a pause for the person's speech resumes, is interrupted or ends within 4 s beyond the endpointing delay of the person going quiet (section 13) | 0 / 0 / 0 (new 2026-10-26) |
| P1 | `first_audio_is_prompt` | once a speech may play, its first audio reaches the speaker within 100 ms of the product having it (the voice's first chunk after its calibrated time to first audio, the realtime server's first delta), unless the speech was paused, cut, its output replaced or the session closing (section 14) | 0 / 0 / 0 (new 2026-10-26) |
| P5 | `idle_is_cheap` | a call wakes its thread at most 200 times per second of virtual time, averaged over the call (section 14) | 0 / 0 / 0 (new 2026-10-26) |
| P4 | `items_carry_their_payload` | a pipeline reply names the models that timed it, a transcribed user turn the STT; a step cut short is recorded interrupted; items and metrics are stamped on the wall clock within the session; metrics carry their label, non-negative durations and their speech id (section 13) | 0 / 0 / 0 (new 2026-10-26) |
| P2 | `realtime_reply_cancelled_with_cause` | the client never cancels the realtime server's reply to the person's turn before it heard of it, nor in the very moment it heard of it with nothing happening (section 13.4) | 0 / 0 / 0 (new 2026-10-26) |
| P2 | `realtime_reply_reads_tool_output` | a realtime reply asked once a tool's results are committed is asked after the tool's output reached the server (section 13.4) | 0 / 0 / 0 (new 2026-10-26) |
| P5 | `realtime_server_accepts_requests` | the realtime server refuses no frame the client had already been told was invalid; a refusal racing an event on the wire is the protocol's (section 13.4) | 0 / 0 / 0 (new 2026-10-26) |
| P5 | soak: residue | after a long call and release, nothing opened is still live | red (2026-10-25): seed 6 leaves a floor hold. Every call must reach its close: one that stopped before it fails the soak |

Every rule is exercised. Promises with no rule are listed per subsystem in section 5.

## 4. Operation x phase

Seeds (of 1,024) in which each input landed in each phase. `#` 20 or more, `+` 1 to 19, `.`
never. Exact numbers are in the appendix.

**Reading it.** For the 40 developer API inputs (480 cells): 427 well covered, 53 rare, none
never. The start column is gone (section 2.2); with it the table had 8 never at `edc90fe24`, all
at start, 24 with the earlier two separate modes, and 98 to 136 with conversations alone.
Most rare cells are in the turn hook column.

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
5. **Four inputs have no host call** and are sent as their real sender's command (section 6).
6. **Handoffs and tasks are not journalled**, so neither a column nor a rule can see them;
   tool-driven handoffs are unreachable (5.5, 5.6).
7. **Hooks without deadlines** may block a run forever (5.6).
8. **Preemptive generation is never off**; batch EOT, explicit VAD and realtime-LLM detection
   are never set up (5.3).
9. **The error budget** (count, threshold, reset) has no rule (5.7).
10. **Realtime tool calls and a realtime model with a separate TTS** are not modelled (2.3).

## 6. Through the public API

The fuzzer drives the core as its host does. `Run::start` (`src/drive/start.rs`) assembles the
session with `SessionApi::host_new`, the `VoiceSession(...)` `lk-agents-py` binds, over a
`FakeRoom` from `lk-agents-rtc` and the fakes, with the setup as `SessionOptions` kwargs, and
awaits `start()`. Every input that is a host call (`src/drive/apply.rs`) is that method with its
argument shapes: `say`/`say_stream`/`say_audio` and their pushers, `generate_reply`,
`interrupt`, `SpeechHandle.interrupt` and `.allow_interruptions`, `commit_user_turn`,
`wait_for_idle(_and_hold)`, `update_agent`, `enter_task`, `update_turn_options`, `drain`,
`aclose` then `close_io`, the `hook_*` answers, `filler_*`, `tool_update`, `dtmf_*`, `ivr_*`,
`attach_amd`, `amd_execute`/`amd_close`, `push_video`, the io setters. Tools run through a
`HostToolRunner` answering the marshal's report. People join and leave the fake room, and the
real room io decides what the session hears of it.

Observation is a test-only probe on `SessionApi` (feature `dst`, enabled by this crate only): it
sees every journal step as it is written and every host event, typed, as the drain takes it, and
hands the session's cores to the run, which stays their queue owner (so the queue policies,
the panic catch and the livelock stop are unchanged). The host's own `on_event` sink still gets
every wire record.

Inputs with no host call, sent as the command their real sender sends:

- `PauseAuthorization`/`ResumeAuthorization`: the AMD detector's own hold (`Origin::Amd`).
- `CommitTurn`: the commit `commit_user_turn` makes after its flush and final wait.
- `Configure` for interruptions, min interruption words, user away and max tool steps:
  `update_options` takes endpointing and turn detection only.
- `Handoff` with `pause_previous` or `resume`: `update_agent` always closes and starts; pausing
  and resuming are an inline task's, which the driver does itself.

What the switch changed in what a run can reach: the backchannel threshold left the setup
(`host_new` never overrides the end-of-turn thresholds, so no host can set one). A tool body's
`AfterIdle` is `async with ctx.foreground()`, `wait_for_idle_and_hold` from the tool's own
origin as `RunContext` sends it, so the tool-origin idle ask (`IdleAskWithdrawn`) is reached. A
`Reply` resolves a dropped call as the host sees it (`interrupt` as done, `wait_for_idle` as
closing, `commit_user_turn` as superseded, `drain` as closing, a DTMF collection as stopped);
only a speech handle the loop dropped is still `Lost`. `AwaitSpeech` awaits the `SpeechHandle`
the host holds (`wait_for_generation`, then `await handle`) and reads it back.

The legs a seed runs on, drawn per seed (`search/setup.rs`):

- **The host's legs** (the fakes): every frame and sink call is tagged, so the conservation
  rules see exactly what went where. A third of these seeds run as **`lk agent console`**: the
  console's socket (`ConsoleIo`, over an in-process stream) is the microphone and the speaker, a
  the session's remote session host on it forwards the session's events and answers the CLI's
  requests through a port that answers as the host would, and a fake CLI on the other end sends
  the microphone and the requests and acks each flushed segment once it has played on the clock.
- **The room's own legs** (half the seeds), as `agent_session.py` runs by default when the host
  hands none: the person joins the fake room and speaks into a microphone track (with a
  pre-connect buffer sent, never sent, or malformed), types in the chat, and the agent's voice
  and transcript go out on the room's published track and transcription streams, which play on
  the clock. Half of these also run the `BackgroundAudioPlayer` beside the session (an ambient
  stream, the thinking sound on agent state), and a third stream the agent's voice to an avatar
  worker instead: `DataStreamAudioOutput` as `session.output.audio`, and the real
  `DataStreamAudioReceiver` in the worker's own room with the runner's playback bookkeeping,
  the two rooms relayed by the run. The relay crosses at the run's next look, not in
  milliseconds, so these seeds run without the clear-buffer give-up timer.

Also drawn: `tts_text_transforms` and the default `VoiceActivityVideoSampler`.

Drawn last in the setup, so a seed keeps the setup it had before they existed (2026-10-26):

- **AMD as `livekit.agents.AMD` attaches it**: `interrupt_on_machine`, `ivr_detection`,
  `wait_until_finished`, a recogniser of its own or the session's finals, and the callee it
  awaits in the fake room (their track: listening at once; a SIP call the case answers later,
  `Amd { Answer }`, or that hangs up; nobody, unless they join within the track wait:
  `participant_missing`). The developer runs
  `execute()` on the verdict (the IVR hand-off on `machine-ivr`) and leaves the block.
- **Keyterm detection** with the LLM the host resolves, on a recogniser that takes keyterms or
  not; the model answers the passes with `record_keyterms`.
- **`RoomOptions(video_input=True)`** on the room's legs: the person publishes a camera the case
  feeds, which the host's own camera would win over.
- **Node factories**: the agents override `llm_node`, `tts_node`, `stt_node` and
  `transcription_node` as the defaults do, over the session's providers; the opening agent's set is
  drawn and every agent a handoff installs flips some of it by its name, and the handoff gives the
  legs again, as `_frame` does. Each opening is recorded with its agent.

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

Re-measured at `049e7e332`, after a seed became the whole of its run (setup drawn from the seed
alone, no learning across seeds), the kept search reaches 158 variants and 4,291 edges in 23 s
on seeds 0 to 1023, with 2 developer cells never reached of 540 (the 45 developer and lifecycle
rows of section 4, the start column gone, section 2.2): `FillerStop` and `ReleaseIdleHold`
during the turn hook. At `56e265f06` it was 159 variants, 4,322 edges in 38 s and none never
reached. `Emitted::BackchannelConfirmed`, reached by the 1e-1 to 1 decade then, is not reached
in this window. The edges fell from 4,499 across the core's rewrite since `edc90fe24`
(synchronous step, providers and outputs by id, and the queue policy of section 8); the causes
were not measured apart.

By decade of randomness, at 1,024 seeds (`049e7e332`):

| Uniform moves per conversation move | Seeds | Wall | Variants | Edges | Cells | Reached only here |
|---|--:|--:|--:|--:|--:|---|
| 1e-4 to 1e-3 | 143 | 5.0 s | 150 | 1,962 | 649 | |
| 1e-3 to 1e-2 | 152 | 5.2 s | 153 | 2,020 | 719 | |
| 1e-2 to 1e-1 | 146 | 3.8 s | 158 | 2,372 | 810 | |
| 1e-1 to 1 | 143 | 1.6 s | 152 | 2,539 | 836 | |
| 1 to 10 | 148 | 1.3 s | 151 | 2,479 | 821 | |
| 10 to 100 | 149 | 1.9 s | 152 | 2,479 | 797 | |
| 100 to 1,000 | 143 | 3.6 s | 154 | 2,857 | 870 | |

By queue policy (section 8), drawn per seed: `Arrival` 508 seeds, 153 variants, 3,508 edges, 14
findings; `Pos` 516 seeds, 158 variants, 3,747 edges, 15 findings. Five variants are reached
only under `Pos`: `Decided::InterruptByAudioActivity`, `Emitted::OverlapInterruption`,
`Input::LegItemDiscarded`, `Tool::ReplyDone`, `Tool::ReplyScheduled`.

What this says:

- Neither end is enough. Before the person listened properly, only near-pure conversations
  reached barge-ins and backchannels; now every decade does, and the middle and the uniform end
  reach the most edges and (call x phase) cells. At `56e265f06` one decade reached a variant no
  other did; at `049e7e332` none does, but the decades still differ in edges and cells, so no
  decade is dropped and none is weighted.
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
- **Fail a livelock by name** (gosim's "no progress" watchdog): a run that takes
  `NO_PROGRESS_STEPS` (20 x the busiest instant measured, 6,262 loop steps) steps at one virtual instant, in the journal or in its own loop,
  is stopped there and reported as `virtual_time_advances`. Before, such a run hung the whole
  campaign.
- **Judge by known bugs, with the protocol fixed before the run** (gosim's benchmark,
  agent-store's held-out findings): `bench/known_bugs.sh`, section 12.
- **Bursts and same-instant orderings** (gosim's scheduling policy): the case owns the driver's
  queue; a quarter of the seeds deliver in arrival order, a quarter by BasicPOS, half by BasicPOS
  with the transitional task bridge (`drive::queue::Policy`); section 8.

## 8. Biases left

Each is a choice the search makes for every run that a real call does not.

- **The order is the queue's, not the runtime's.** Applying an input does not yield, so a case
  can build a burst the driver untangles in one cycle, and the case picks what the driver's queue
  delivers next: arrival order, or basic partial-order sampling over the event sources (the
  microphone's frames one of them) and the driver's task pool, or, under the transitional task
  bridge, each task the run spawns as its own process (`drive::queue::Policy`). Each source keeps
  its own order. There are no read/write footprints and no race-based reassignment (owner
  decision 21). The pool's tasks still run in tokio's fixed order on a single thread.
- **Time moves in 100 ms beats of 10 ms frames.** Frames never arrive one by one, and nothing
  lands inside a beat.
- **Fixed pools of values**: the person's sentences (`USER_TURNS`) and the agent's lines
  (`AGENT_TURNS`) are short fixed lists, and stock delays come from one list (`DELAYS_MS`).
- **Uniform over rows**: a uniform move weights every row the same, so an actor with more rows
  gets more moves (the developer has about 45 rows, the room one).
- **The setup is fixed where section 2.3 says so**: batch end-of-turn, explicit detection modes,
  AMD and interruption options, close reasons.
- **Run length**: every run has the same budget (3,000 inputs, about the first minute and a half
  of a call whose median is five minutes); short calls only happen when the call ends early. A
  per-seed budget would cover both.
- **One end for runs that outlast their budget**: those all end with the same release and close.
- **One person, one microphone**, and the person always starts the conversation.
- **The person is content-blind**: whether they barge in or backchannel does not depend on what
  the agent says, and what they say is one of nine sentences, never an answer to the agent.
- **The setup is uniform**: every switch is a coin, whatever earlier seeds found. Learning
  across seeds would make a seed's run depend on every seed before it in its window, so a
  finding could not be rerun from its seed alone.

## 9. How to read the numbers honestly

- A journal variant or a code line reached is not a guarantee checked. 158 of 160 journal
  variants are reached (missing: `Emitted::BackchannelConfirmed`, `Tool::SessionToolsExtended`),
  but section 5 lists dozens of promises nobody checks on them.
- Most seeds fail at least one rule (838 of 1,024), including conversation seeds, the two least
  random decades (166 of 295). Every failure is of a contract in `check::KNOWN_FINDINGS`, and
  every contract made at least one expectation. Until the failures are triaged, the driver's
  defect rate is unknown; that is the number a ship decision needs.
- The widest clusters are `speech_done_once` (559 seeds), `output_replacement_is_clean` (434),
  `interruption_has_cause` (228) and `interrupted_handle_is_journalled` (219);
  `forwarded_while_enabled`, audio not reaching a recogniser, is down to 96 (317 at
  `56e265f06`). The soak counts each call's finals: 4 to 103 per call, three calls under ten
  (seeds 0, 1 and 6; seed 0 over 12,000 inputs). The soak does not judge that count, only
  leaks; `tests/replaced_inputs.rs` fixed the fakes' part of it (a replaced recogniser kept
  being spoken into), and what is left is still to triage.
- Registered providers are checked to be released: `every_resource_released` requires the
  session's resource table (`lk_agents_abi::resources`) to be empty once the driver exited, and
  it is not a known finding, so it held on every seed of the survey.
- A stuck agent is caught by `speech_is_heard` (15 s of dead air with nothing to wait on) and a
  stuck pause by `pause_resolves` (section 13). Both bounds are generous on purpose: they catch an
  agent that stays silent, not one that answers slowly; latency has no rule.
- No finite campaign shows an interaction is impossible. A `.` cell is a hole in the search; a
  `#` cell is a place the search goes, not proof the behaviour there is right.
- Detection rate is unmeasured: known bugs have never been held out to check the search finds
  them.

## 10. Code reached

Line coverage of `lk-agents-voice` over every `lk-agents-fuzz` test, the 1,024-seed survey and
the soak included (2026-10-26, after AMD, keyterm detection, the room's camera and the node
factories were wired and drawn): **83.6% of 23,634 lines** (79.6% of 23,544 just before; 64.8% of
25,678 before the room's
legs, the console, the transforms and the players were drawn; 65.2% of 25,412 right after the
switch to the public API).

**Production-reachable: 87.0%** (19,639 of 22,583 lines): the same measurement over the lines a
deployed Python agent can run. Left out, each for its reason below: `workflows/` (owned by the
text-mode rework), the remote-session client and `room_io/rooms.rs`. Before the wiring this read
85.1% of 21,164 lines, a total that also left out `amd/` and the keyterm LLM pass because
`livekit-agents-next` never ran them; counted the new way it was 82.9% (18,644 of 22,493).

| Area | Lines | Reached | Note |
|---|--:|--:|---|
| `audio_recognition/` | 1,512 | 94% | hearing: VAD, STT, end of turn, overlap |
| `endpointing.rs` | 156 | 99% | (was 58%: the unused host classes went) |
| `ivr/` | 225 | 97% | |
| `room_io/_output*` | 787 | 91% | the agent's track and transcription streams (was 28%) |
| `tool_executor.rs` | 558 | 91% | |
| `room_io/room_io.rs`, `link.rs` | 464 | 89% | |
| `generation/` | 2,156 | 89% | |
| `agent_activity/` | 5,401 | 88% | |
| `remote_session/console.rs` | 114 | 89% | the console's socket (was 0%) |
| `transcription/text_transforms.rs`, `filters.rs` | 450 | 84% | (was 0%) |
| `transcription/synchronizer.rs` | 590 | 86% | |
| `agent_session/` but the API | 2,735 | 86% | |
| `agent_session/api*` | 1,297 | 79% | |
| `room_io/_pre_connect_audio.rs` | 141 | 78% | (was 0%) |
| `background_audio.rs` | 569 | 71% | the player, beside room-legs seeds |
| `speech_handle.rs` | 587 | 70% | |
| `avatar.rs` | 561 | 72% | sink and worker receiver |
| `io.rs` | 174 | 68% | the node chain seams |
| `room_io/_input.rs` | 373 | 82% | the person's microphone and camera tracks (65% before the camera was drawn) |
| `amd/` | 914 | 90% | (was 74%, unwired) |
| `keyterm_detection.rs` | 567 | 92% | the LLM pass (was 24%, unwired) |
| `agent_session/nodes.rs` | 107 | 78% | (was 11%) |
| `remote_session.rs`, `remote_session/driver.rs` | 935 | 49% | the host side runs; the client does not (below) |
| the rest | 1,716 | 86% | |

The justified gaps, outside the production-reachable figure:

- `workflows/` (733 lines, 15%): what the text-mode rework left (result types, validation
  helpers); owned by that work.
- The remote-session client (`ClientRuntime`, 180 lines, and its `event_name`, 17): the far
  end of the protocol, bound for Node and Python clients, never run by an agent.
- `room_io/rooms.rs` (121, 0%): `_core.FakeRoom`, a Python test fixture, and `OfflineRoom`, a
  constructor on the wall clock.

Production-reachable code still unreached, by size: the session API's rarer host calls
(`set_participant`, `resolved_options`, the AMD and DTMF accessors), `speech_handle.rs`'s
`hold_interruptions` and the host's `interrupt`, the remote-session encoders of the events no
seed emits (agent config updates), the room's session transport, and the
avatar sink's interrupted path after a give-up timer.

## 11. Outside this suite

Measured nowhere here, and each a separate source of production risk:

- The Python voice pipeline (`livekit-agents/livekit/agents/voice/`), which is what ships today:
  `docs/internals/rust-core.md` says it does not call this driver.
- The binding from Python to the driver (`lk-agents-py`), plugins reached through host proxies.
- Threads, lanes and leases in agent-db, the A2A text door, and a text turn landing on a lane
  while a call is live (`storage/live.py`).
- Real provider behaviour beyond what was recorded: section 13 lists what each fake is
  calibrated from and what it still cannot do.
- `lk-agents-inference` (native gateway clients, fallback adapters), `lk-agents-media`,
  `lk-agents-rtc`.

## 12. Keeping it true

**The launch gate** is one command, `rust/crates/lk-agents-fuzz/launch-gate.sh`, and the merge gate
in CI (`.github/workflows/launch-gate.yml`: 4,096 campaign seeds on every pull request and push to
v2.0, 32,768 nightly). It prints a coverage table placing every workspace crate as executed,
compiled only or outside (with the job that covers it), and fails on a crate in no list; then the
workspace build, the voice core's unit tests, the provider clients' unit tests
(`lk-agents-inference`, `lk-agents-abi`, and abi's `python` feature under nextest), the fuzz suites
(the scheduler's small programs, every contract's examples, counterexamples and probe, determinism,
the regression tests), the 1,024-seed survey, a campaign on the next window (`VOICE_GATE_SEEDS`,
32,768 by default) and the soak (`VOICE_SOAK_SEEDS`, 8). Every verdict is a runner's exit status:
the campaigns run under nextest, which fails a selection that runs no test, and zero seeds are
refused. It fails on any finding outside `src/known_findings.tsv` (open findings keyed by
contract and subject kind; CI rejects an added line), on a soak residue, and, nightly
(`VOICE_FULL_WINDOW=1`), on a known finding the full window no longer reaches once its fixing
regression exists. Each campaign prints its boundary (`src/boundary.rs`): the cores it steps, the
fakes, what still runs as tasks, and the runtime adapters left to K-30's verification. About five
minutes on a laptop (the 32,768-seed campaign is three). Not in the gate yet (K-12 closure):
`pytest --unit --audio_eot`, the binding conformance suite, backend contracts, IPC stress and the
runtime verification suites.

**Reproducing a finding.** A seed is the whole of a run: its setup, its conversation, its
providers' behaviour and its schedule are all drawn from it, and nothing carries over from the
seeds before it. Every finding the survey reports comes with its seed and the line that reruns
it alone:

```bash
cd rust
VOICE_SEED=4372 cargo test -q --release -p lk-agents-fuzz --test survey -- --ignored --nocapture
```

which prints that run's setup, every failure in full and the journal variants it reached;
`VOICE_TRACE=1` adds every input as it is applied. The seed reproduces its run on the same code
only: once the finding is understood, it becomes a hand-written regression test in `tests/regressions/`
that reaches the bug through the fewest inputs, fails before the fix and passes after it, and
holds whatever the search later changes.

What it can and cannot miss:

- **Verdicts never depend on real time.** A livelock is counted in steps at one virtual instant
  (`NO_PROGRESS_STEPS`, twenty times the busiest instant measured), in the journal and in the
  run's own loop, and reported as `virtual_time_advances`. Every wait the harness makes is
  bounded in virtual time (`BUDGET_MS`), except the wait for a begun close, which runs until the
  session exits (a virtual day without an exit is `close_finishes`), and every loop in rounds
  (release, join), so a slow machine cannot change a verdict. Every run that reaches its ending
  is judged closed or reported.
- **A seed reproduces its run.** `tests/determinism.rs` checks that one seed explores one case
  and that a case leaves the same recording twice.
- **A real hang ends the process, loudly.** A product task that never yields blocks the run's
  only thread, where no step count can see it; a watchdog thread beside it (`VOICE_HANG_S`, 600 s)
  aborts with the seed. It is a crash, not a verdict: rerun it with `VOICE_SEED=<seed>
  VOICE_TRACE=1`.
- **Leaks**: tasks (`every_task_finishes`, with the spawning line), speeches (`speech_settles`),
  answers (`api_calls_return`), provider streams (`provider_streams_closed`), turn-owned work
  (`nothing_left_live`), the provider table (`every_resource_released`), and growth over forty
  turns (the soak).

```bash
cd rust
# reach, failures and the operation x phase table, by decade of randomness (about 25 s)
VOICE_CELLS=1 VOICE_SEEDS=1024 cargo test --release -p lk-agents-fuzz --test survey -- --ignored --nocapture
# long calls, and what is still live after them
cargo test --release -p lk-agents-fuzz --test soak -- --ignored --nocapture
# code reached (a few minutes; needs cargo-llvm-cov and the llvm-tools component; exits non-zero
# while the soak is red, the profiles are still written)
VOICE_SEEDS=1024 cargo llvm-cov --release -p lk-agents-fuzz --no-fail-fast --no-report -- --include-ignored
cargo llvm-cov report --release --summary-only | grep lk-agents-voice/src
```

Next:

- **The known-bug benchmark** (`bench/known_bugs.sh`) reintroduces each fixed bug in a copy of
  the tree and measures every queue policy's hit ratio on matched seeds; a change to `drive/` or
  `fakes/` records its `--compare` against the commit before it (K-14).
- **Generated**: add the phase columns of section 5, have each rule register the (promise,
  operation, phase) cells it judges, write sections 3 and 4 from what the survey observed, and
  fail a test when this file is stale (agent-store's `dst/COVERAGE.md` and `TestCoverageMatrix`).

## 13. Fidelity and liveness

Two real-room bugs passed this suite: a realtime barge-in that left the agent silent
(c6881abb9) and chat items that lost their models (9024425d9). Both came from the same gaps: the
fakes produced orders and timings no real provider produces, and no rule judged silence or a
payload. This section is what closed them.

### 13.1 What the fakes are calibrated from

`corpus/record.py` records LiveKit Inference traffic as event kinds and relative timing, never
what anybody said; `corpus/derive.py` turns the recordings into `corpus/calibration.json`
(transition counts between event kinds, delay samples, shares), and `src/corpus.rs` draws from
those samples per seed. Re-recording needs LiveKit Inference credentials: run `record.py`, then
`derive.py`, and commit both the recordings and the calibration.

| Fake | Calibrated from | Still cannot do |
|---|---|---|
| Realtime (`fakes/realtime/`) | The production realtime core (`lk-agents-inference`: the conversation machine and the connection policy, reconnects and all) driven by a pump that mirrors the runtime's (`lk_agents_runtime::realtime::session`: its biased order, the writer's 100 ms audio batches and response receipts), over a stateful simulation of the OpenAI Realtime server in the protocol's typed records (`Outgoing` in, `Incoming` out; section 13.4); the fallback adapter's contract over two of them on the inference fallback policy; held to the real server's answers to the protocol's edge cases (`corpus/probes/`, `corpus/probe_realtime.py`) and, where the probes are silent, the docs. Recorded on `gpt-realtime` (turns, barge-in, backchannel, silence, client cancel, tool call, manual turns; server and semantic VAD; two takes each): speech start and stop lag, response creation after the stop and after `response.create`, item and first-audio latency, delta sizes, done after the last audio, input transcription after commit and its deltas per word, cancel lag, audio per character, the share of responses with function calls and rate limits, error latency; from the probes, the end of a cancelled reply after the cancel and the refusal latency | Other dialects (Gemini, xAI, Azure differences); images; the server's VAD does not judge the audio (it needs some, and the case says when it fires); what a reply says does not depend on what it read (only its usage does); session expiry |
| Duplex (`fakes/realtime/duplex.rs`) | The inference duplex core (`Segmenter`, `FixedGate`) driven as the runtime's duplex session drives it, over a model that speaks when asked (an ask while it speaks is answered by its next utterance): 100 ms frames at playback pace, its own silence after, span-stamped word fragments just ahead of their sound, and in 35% of utterances the last words' text after the sound ended (`TranscriptAttribution`); 15% of realtime setups, from the model's own dice | Tool calls, barge-in and input transcription (the model ignores what it hears); an adaptive gate; latencies not recorded from a real duplex provider |
| Recogniser (`search/conversation.rs`) | Per seed one of Deepgram nova-3, Deepgram Flux, AssemblyAI universal streaming: first words late, start of speech with them, usage before each final, Deepgram's cuts and empty finals, AssemblyAI's second final, Flux's preflights | Confidences and word timings from data; language detection; diarization; the real sockets' reconnects |
| LLM (`fakes/llm.rs`, `search/conversation.rs`) | `gpt-4.1-mini` streaming: the role on the first chunk, token chunks, a finish chunk, a usage chunk; each request answered on the model's own clock, the first chunk after the recorded delay, then pieces of the recorded size at the recorded gaps, failing anywhere in the answer | Other providers' chunk shapes; refusals and content filters; parallel tool calls, a preamble before a call, streamed arguments, real usage numbers |
| TTS (`fakes/tts.rs`) | Cartesia `sonic-3`: time to first audio; audio as long as the text takes at the recorded milliseconds a word (drawn per stream), in frames of at most 100 ms | The recorded chunking and gaps; failures shaped like a real provider's (they are injected uniformly) |
| VAD, room | Not calibrated: VAD probabilities are synthetic, the room is `FakeRtc` | Silero's real output; network loss and jitter |

Every provider event the core consumes (`SpeechEventType`, `VadEventType`, the realtime wire and
events) is a `Vocabulary` (`fakes/vocabulary.rs`): a variant no fake covers or explicitly skips
is a compile error. `tests/conformance.rs` replays the recordings through the fakes and fails on
an event kind, a response or utterance shape, a documented ordering or a delay the real provider
never showed.

### 13.2 Liveness and payload rules

- `speech_is_heard`: from its creation, and again from when the scheduler let it go, a speech
  pushes audio within 15 s, ends, or still waits on something that holds (a provider, tool, hook
  or stream the case owes; authorization; the floor held by a playing speech, or one paused for
  the person within `pause_resolves`' allowance; the developer's hold; the gap between speeches;
  the person speaking; nobody in the room; a provider failing). A speech that ended unheard
  (interrupted, or its realtime response cancelled) was cut by the person, the host, the server
  or the close. With c6881abb9 reverted it fails survey seeds 405 and 661 and
  `realtime_onset_reply_heard`.
- `pause_resolves`: a pause for the person's speech resumes, is interrupted or ends within 4 s
  beyond the endpointing delay in effect, counted from the first stretch the person stays quiet.
  It does not catch c6881abb9 in the search (the person's next turn resolves the pause there);
  it caught the late-verdict pause below (6 of 2,048 survey seeds with that fix reverted).
- `items_carry_their_payload`: see section 3. With 9024425d9 reverted it fails 245 of the
  1,024 survey seeds (170 replies naming no LLM, 75 user turns naming no STT) and
  `turn_metrics_name_models`.

Found by them and fixed, each with a regression test that fails with its fix reverted:

| Defect | Test |
|---|---|
| Work the waiter pass started (a background tool's deferred reply, a filler's `say`) was scheduled only at the next unrelated wake: up to 15 s of dead air | `tool_reply_scheduled_at_once` |
| A pause decided after the person stopped (a detector's late verdict) never armed its resume | `late_verdict_pause_resumes` |
| The native realtime session stamped its metrics with seconds since the model was created and its items in 1970 | `realtime_metrics_wall_clock` (500 of 1,024 survey seeds with the fix reverted) |

`livekit-agents-next/scripts/parity.py` (deleted since, owner decision 22) compared payloads whole against the reference: each
message's metrics (the models it names, which timings it carries), each metrics record's kind,
label, model and attribution, and the history's order by `created_at`. It found, and the core now
matches the reference on: a `say` message naming the LLM and TTS and an `llm_node_ttfs`; a realtime
message carrying an `llm_node_ttft`; a realtime turn's EOUMetrics naming no speech; a turn asking
for no reply (`skip_reply`, or one AMD consumed) run through `on_user_turn_completed`
(`skip_reply_turn_no_hook`). Five differences are listed in its `KNOWN` with their reasons:
`eou_names_resolved_detector`, `fallback_stt_identity_at_bind`, `node_str_ttft`,
`own_audio_tts_node_ttfb`, `keyterm_pass_unattributed`.

The calibration itself found four more (c5be2a811): a realtime reply whose audio was cut
reported its tools not running; a background tool's deferred reply on a realtime session was
built as a pipeline reply; realtime tool follow-ups carried no `tool_choice`; a voiceless filler
with interval 0 fired forever in one instant.

Open: on a realtime session a filler's `say` speaks with no voice instead of being refused.

### 13.3 Cost

The survey (1,024 seeds) runs in 1.7 s and the campaign (4,096 seeds) in 6.3 s on a laptop,
within the 2 s and 11 s they took before the calibration. With the realtime server of 13.4 they
take 1.2 to 1.4 s and 5.4 to 5.6 s.

### 13.4 The realtime server and its probes

`corpus/probe_realtime.py` drives the protocol's corners on the real server, through LiveKit
Inference (`openai/gpt-realtime`, the `lk` project's credentials, as `record.py` does), and
keeps each session's client frames and server events as a fixture in `corpus/probes/` (content
as its length, ids as ordinals): every frame a server might refuse on an idle session, a reply
asked over an active one, a cancel mid-audio and one in the same breath as a new reply,
truncations past, at and inside a reply's audio, a reply asked before and after a tool's output
and in the same breath as it, instructions and the voice changed mid-reply and after, server VAD
told not to interrupt and not to reply, manual turns with a commit mid-reply and a clear, barge-in
on both VADs over a reply still generating, replies back to back, and a socket closed mid-reply.
`tests/conformance.rs` replays every fixture through the fake server frame by frame, each recorded client frame as the typed frame the realtime core sends for it (one the core never sends -- `conversation.item.retrieve`, `output_audio_buffer.clear`, an assistant audio part -- is left out, and a frame the core sends without an `event_id` is read as cited by nothing) (a frame the
probe sent on an event goes once the fake sent that event): each frame must get the real
reaction (the acknowledgement or the error code, citing the frame or not, within the real delay's
envelope), each onset over a reply must end it as the real one did, and the events must keep the
documented order and the recorded response shapes.

What the fake holds and enforces, from the probes:

- **The conversation**: items by id and kind, placed where the server places them (a reply's
  item when the server adds it, after a client item that arrived first), deleted, truncated (a
  truncation cuts what the item holds; a later one is judged against the cut, within a
  millisecond of rounding). A reply reads what the conversation holds when asked (its usage says
  how much).
- **One reply at a time**: `response.create` over an active one is refused
  (`conversation_already_has_active_response`); a cancel frees the conversation at once, so a
  reply asked in the same breath is created before the cancelled one's last audio delta and
  `response.done`; a cancel with nothing active is `response_cancel_not_active`, naming an unknown
  response `response_cancel_not_found`; a cancelled reply's call closes `incomplete` and stays
  among its outputs.
- **The refusals and their exact codes**: `input_audio_buffer_commit_empty` under 100 ms,
  `item_create_duplicate_item_id`, `item_create_invalid_previous_item_id`, `invalid_tool_call_id`
  (an output for a call the conversation does not hold), `missing_required_parameter` (assistant
  audio content), `item_delete_invalid_item_id`, `item_retrieve_invalid_item_id`,
  `item_truncate_invalid_item_id`, `unsupported_content_type` (truncating a user item),
  `invalid_value` (past the audio, or an unknown client event, `output_audio_buffer.clear`
  included), `cannot_update_voice` once audio was sent; some cite the frame, some do not, as the
  real ones do. The server answers the client's frames in order.
- **Its VAD over the audio sent**: an onset needs audio appended since the buffer was last
  emptied; with `interrupt_response` it cancels the reply (`turn_detected`), without it the reply
  completes and the turn is committed after it with no reply of its own.

Every refusal, and two answers the real server gives from the wrong state, are recorded as a
`Breach`, judged by three rules:

- `realtime_server_accepts_requests`: no refusal the client knew better than to provoke (it had
  been told the response it cancels was over, the one it asks over had been created, it emptied
  the buffer itself; an item, truncation or event the server never takes). A refusal racing an
  event on the wire is the protocol's.
- `realtime_reply_reads_tool_output`: a reply asked once a tool's results are committed, while the
  conversation ends in its call, is followed in the same breath by the output and by no other
  reply: it read the call without its result. With the outputs pushed after the follow-up's
  `response.create` (the current equivalent of 8a4f95729's bug) it fails 21 of the 1,024 survey
  seeds and 76 of the 4,096 campaign seeds.
- `realtime_reply_cancelled_with_cause`: a client cancel stops the server's reply to the person's
  turn before any audio and with no onset told over it, in the moment the client heard of it with
  no input, no time passing and no refusal to schedule its speech. With c6881abb9 reverted it
  fails 5 of the 1,024 survey seeds and 21 of the 4,096 campaign seeds.

Where the docs and the server disagree, the server wins: the docs say a cancel with nothing in
progress is safe, the server refuses it; they give no minimum for a commit, the server wants
100 ms; they say nothing of a reply asked right after a cancel, the server takes it and creates
it before the cancelled reply ends; a truncation one millisecond past the audio is taken; a
function call cut by a cancel is still among the reply's outputs, `incomplete`; the
`previous_item_id` of an assistant message's `conversation.item.done` is sometimes `null`.

Found by it and fixed, each with a regression test that fails with its fix reverted:

| Defect | Test |
|---|---|
| On a reconnect the driver replayed its conversation onto the new connection after the session had replayed it itself: every item went twice, the server refused each second one | `realtime_reconnect` |
| The native session counted the audio sent before a reconnect as still in the server's buffer, so a commit after it named nothing and was refused | `realtime_reconnect` |
| A cancelled reply's late `response.done` closed the reply asked after it, whose audio was dropped, and its late deltas were played as the new reply's | `realtime_cancelled_tail` |
| A reply asked while the server generated one (the host's `generate_reply` or `say`, a deferred tool reply, the tools' follow-up), or crossing a reply the server started on its own, was refused and lost (the reference asks the same and is refused the same). Replies now go one at a time, as speeches are scheduled: a request the server would refuse waits in a queue and goes once the reply it would cross is done, nothing cut; one refused for a reply that crossed it on the wire goes again after it | `realtime_reply_queued` |
| The outputs reached the server only on the step's later wake, after the commit had let a waiting reply go, which read the call without its result (campaign seed 1978) | `realtime_outputs_before_next_reply` |
| A tool output for a call the developer's `update_chat_ctx` removed while it ran was sent, naming a call the server no longer held (`invalid_tool_call_id`); a later edit re-sent it too. It is not sent, and an edit leaves out an output whose call neither side holds; a truncation of a message an edit removed is not sent either | `realtime_context_edits` |
| A context update sent while the previous one was unconfirmed was diffed against what the server had confirmed, and deleted again what that one deleted (`item_delete_invalid_item_id`); it is now diffed against what the server is being brought to | `realtime_context_edits` |
| A reconnect that crossed an unconfirmed context update replayed the confirmed conversation and then that update's frames, creating twice what it deletes and creates again; an update right after any replay was diffed against nothing (`item_create_duplicate_item_id`). The new connection gets the conversation the server was being brought to, tracked until confirmed | `realtime_reconnect` |

None is open: `realtime_server_accepts_requests` is not a known finding.

## 14. Manually found bugs and what catches them

Would the automated checks have caught every bug found by hand? Not before this section. Of the
55 bugs below, fixed on 2026-10-25 and 2026-10-26 after real-room calls, latency runs, parity
diffs, smoke runs, coverage work and review rather than a failing seed, 9 passed every check with
their fix reverted. They fell in classes the checks did not look at: Python glue and job wiring
(the DST runs the core alone; smoke and parity follow fixed scripts), latency (no check measured
time to first audio, and the DST does not run the production provider clients), cost (nothing
counted wake-ups) and helpers no script calls. Each now has a check, marked **new** below, proven
to fail with the bug and to pass without it.

**Method.** Each fix was reverted alone on the tree of 2026-10-26 (`402ac3b49`); where the code
had moved, the bug was re-created in today's code with the smallest change that brings the
behaviour back (the patches are not committed). Then the checks ran: the launch gate's DST steps
(the fuzz suites with the regression tests, the 1,024-seed survey, the 4,096-seed campaign),
`smoke.py`, `parity.py` (since deleted) on a wheel rebuilt with the bug, `fuzz.py` and, for provider clients,
their crates' unit tests. "gate:" names the regression test that fails, "search:" the rule and
seed the random survey or campaign fails on with no hand-written test. A regression test written
with a fix guards that exact path; the search guards its neighbours.

| Bug | Fix | Found by | What catches it now |
|---|---|---|---|
| A session started on the job's room never joined the call: not the job's primary session, no report, not closed on shutdown | 298135e33 | real room | **new** `fuzz.py` job seeds: "the session on the job's room is not the job's primary session" (seeds 1, 3, 4) |
| The primary session no longer served the remote session protocol: every `lk agent simulate` timed out | 8663bd6af | real room | smoke (`console`: no session state) |
| A reply cut before any of its audio played was committed whole: the model read its unheard question as asked | 564b876ed | real room | gate: `unheard_reply_not_committed`, once 519e2f411's guard (which also drops the item) is reverted with it; alone it no longer reproduces |
| A spoken user turn carried no end-of-turn metrics: no reply had an `e2e_latency` | 2ca3177a0 | real room | gate: `user_turn_metrics`, `preemptive_reply_latency` |
| A realtime server's item placeholders went into history empty; the user's words were lost | 64646b925 | real room | gate: `realtime_items_filled` |
| On a realtime model the default turn detector turned the server's turn detection off | f1c8fc3cb | real room | **new** `fuzz.py` realtime seeds with the default VAD: "the realtime server's turn detection was turned off" (seeds 4, 42, 70) |
| The OpenAI realtime model never declared `auto_turn_reply_generation`: the core asked a reply over the server's own (+0.4 s, `conversation_already_has_active_response`) | 361c7c874 | real room | gate search: `speech_is_heard` (campaign seed 1209); re-created in the Rust native model, where the setting now lives |
| A toolset that failed its setup (an unreachable A2A back office) crashed the job before the agent joined | bba142c4c | real room | **new** `fuzz.py` failing-toolset seeds: "raised ConnectionError" (seeds 6, 7) |
| `lk.agent.state` taken before the room connected was never written | 677959027 | real room | gate: `agent_state_written_on_connect` |
| A realtime reply's audio was held until the server finished generating it (0.5 to 1 s) | c8a832860 | real room | parity (`realtime_reply`, `realtime_stt_default_detection`); **new** `fuzz.py` measured replies: "a reply took 1.94 s to be heard, its providers 0.51 s" (seeds 26, 30, 37) |
| A realtime tool's output reached the server after the `response.create` that should read it; updates could land out of order | 1145f5450 | real room | **new** `fuzz.py` realtime measured replies: the reply comes before its user message reached the server (seeds 7, 37, 48) |
| A cut reply fell back to its whole text when a host output reported the cut late (every real room) | 986135c09 | real room | gate: `cut_reply_keeps_heard_words` |
| A realtime generation's words never reached the transcription output | 745680163 | real room | gate: `realtime_transcript_forwarded` |
| A realtime barge-in paused the reply instead of interrupting it: the agent went silent | c6881abb9 | real room rerun | gate: `realtime_onset_reply_heard`, `realtime_server_onset_interrupts`; search: `speech_is_heard` (seeds 405, 1209) |
| Chat items lost their STT/LLM/TTS model metadata | 9024425d9 | real room rerun | gate search: `items_carry_their_payload` (seeds 7, 1035) and its probe |
| A job log record carrying an Agent raised a PicklingError traceback (and hung the transport) | 7f2488fc8 | real room rerun | smoke (`log transport`: times out, smoke never exits) |
| The gateway TTS reconnected and reconfigured per reply (0.57 s to first audio against 0.36 s) | 54c213e74 | latency.py on real calls | **new** gate: `voice_connection_reused` (reply 0: 417 ms against 217 ms owed); **new** gate: its own unit test (`a_second_synthesis_reuses_the_socket_the_first_finished_on`), which ran nowhere until the gate ran the provider clients' tests |
| TTS TTFB and time to first sentence counted from the first token, not from the text sent | 1bbf57ccb | real calls | gate: `tts_first_byte_anchor` |
| `llm_node_tps` counted chunks over the whole request | c55a5f1cc | real calls | gate: `llm_tokens_per_second` |
| An adopted preemptive reply reported no `e2e_latency` | aec87eef0 | real calls | gate: `preemptive_reply_latency` |
| The gateway LLM opened its connection inside the first reply's TTFT (no prewarm) | 000af03c2 | latency investigation | **new** smoke (`llm_prewarm`: the gateway saw no `GET /models`) |
| JobRun regression: the entrypoint, shutdown callbacks and session end ran without the job's context; the room never connected | 75b974c28 | real calls | reverting alone no longer reproduces (the core now awaits those steps inside the job task, which holds the context); **new** `fuzz.py` job seeds fail on the class: "the entrypoint ran without the job's context" (seeds 1, 3, 4) |
| Every `room_options=` raised TypeError | f0170e9c0 | parity wiring | smoke (`room_camera`); **new** `fuzz.py` room-options seeds (seeds 0, 2, 5) |
| After an AMD machine verdict the releasing turn still got a reply | 5bbdb5392 | parity wiring | gate: `amd_releasing_turn_no_reply` |
| `conversation_item_added` omitted the agent handoff items | 73971cd6c | parity | parity |
| Error events carried a dict, not the provider's typed error and source | 97c378710 | parity | parity |
| `stt.FallbackAdapter` made `start()` raise AttributeError | 6ea2177a1 | parity | parity (`stt_fallback`); **new** `fuzz.py` fallback-STT seeds (seeds 5, 10, 12) |
| A reply's speech was announced not user-initiated and after its user item and EOU metrics | 50ea404d7 | parity | parity (`user_turn`, `interruption`, `backchannel`, ...) |
| A turn over an uninterruptible speech still got a reply | fb2f9816e | parity | gate: `uninterruptible_drops_turn`; search: `speech_settles` (seed 2934) |
| A TTS failure before any audio committed an assistant item | 519e2f411 | parity | gate: `unvoiced_synthesis_no_item` |
| Unset turn detection with an STT and no VAD took the STT path, not the reference's VAD path | 1b24973d9 | parity | gate: `unset_detection_stt_only`; search: `speech_settles` (seed 3505) |
| The room transcript arrived all at once at the end, not word by word with the audio | 6a0d4d89c | parity | parity (`live_transcript`, `interruption`, `false_interruption`) |
| After a handoff the new agent saw calls to tools it lacks | 71698a569 | parity | parity (`handoff_update_agent`) |
| A realtime reply's state changed after its message was added | 4f485f3b3 | parity | parity (`realtime_reply`, `realtime_interruption`) |
| Under STT turn detection the STT's end of speech never resumed a false-interruption pause (aclose hung) | d871f977f | parity | gate: `stt_end_resumes_pause` |
| A final over a paused speech planned an extra preemptive request | 3977e266b | parity | gate: `final_over_paused_no_preemptive` |
| `session.input.audio` was not the room's microphone | a109d87c4 | parity | smoke |
| A final transcript's pause never ended while a drain held scheduling | 4a8d5c8a6 | real calls (seed 2068) | gate search: `pause_resolves` (campaign seed 3693) |
| A non-streaming TTS reported one TTSMetrics per reply, not per sentence | 863c3a853 | parity | parity (`say_cut`, `interruption`, `vad_*`, ...) |
| An `llm_node` returning a str was metered | 12fb3f3fa | parity | gate: `node_text_not_metered` |
| `transcription_node` and `realtime_audio_output_node` stayed the opening agent's after a handoff | 7015456df | review | gate: `nodes_are_the_active_agents` (probe; survey seed 43 with the install reverted), smoke |
| A realtime session was not told to stop at a close | f73f4dbd3 | parity | gate: `realtime_close_interrupts` |
| No VADMetrics, and no LLMMetrics for AMD's classification | 36057f580 | parity | gate: `model_metrics_reported` |
| A manual-mode turn with no words was dropped at the close | b706c2c9b | parity | gate: `manual_turn_closed_empty` |
| `commit_user_turn()` committed nothing until awaited | bd872d15a | parity | parity |
| A `tts_node` over the default one reported one record for its output, not the provider's calls | a79ed02a0 | parity | parity (`tts_node`, `tts_node_interruption`) |
| A `tts_node` running no default node was metered | 9b5a5a357 | parity | parity (`tts_node_own_audio`), re-created in the node's call ledger |
| A transcript pause showed a word whose audio had not played | 848f9fb25 | parity | parity (`realtime_interruption`) |
| `tts_text_transforms` never reached the voice: markdown and emoji were spoken | 8255100c9 | coverage | gate: `tts_text_transforms` |
| The room's speaker stayed paused until the close (three ways) | e142b2fbd | coverage | reverting the releases alone no longer reproduces; its regression tests (`pause_outlives_its_speech`) stay, and `speech_is_heard` bounds the silence at 15 s |
| A recogniser failing mid-utterance left the person speaking for the rest of the call | 5121ed4bb | coverage | reverting alone no longer reproduces; its regression test (`recogniser_failed_speaking`) stays |
| The idle background audio mixer polled every millisecond | a61e6faa0 | coverage | **new** gate search: `idle_is_cheap` (survey seed 1: 1,000 wake-ups a second against at most 29) |
| A cancelled `BackgroundAudioPlayer.aclose()` closed the player anyway | d0e624905 | review | smoke (`background_audio`) |
| `AudioStreamDecoder(sample_rate=None)` decoded at 48 kHz, not the source's rate | e04d5310e | review | **new** smoke (`decoder`: `(None, {48000})`) |
| A recognize-only STT failed to open on a call | b37ea9517 | glue round 6 | smoke (`session.run()`: "the session has no a streaming STT"); `fuzz.py` recognize-only seeds (seed 0) |
| A host stream cancelled at the close was reported as `stt_error` carrying a CancelledError | 6689bcd7a | smoke (rare) | **new** `fuzz.py` (run by smoke), a plugin stream ending in the close's cancellation: "Exception in callback STT._core_failed(CancelledError())" (seeds 2, 23, 44); its own abi unit test needs the binding and runs in no gate |

Three fixes no longer reproduce their bug when reverted alone on today's tree, because another
path now covers it: e142b2fbd and 5121ed4bb (the speaker's pause outliving its speech, the
recogniser failing mid-utterance: their regression tests pass with the releases reverted) and
75b974c28 (the core now awaits the job's steps inside the job task, which holds the context, so
the explicit context copy is belt and braces). Their regression tests, and for 75b974c28 the
`fuzz.py` job seeds, guard a recurrence through the original path.

### 14.1 The checks added

- **`first_audio_is_prompt`** (DST rule, a latency contract on virtual time): once a speech may
  play, its first audio reaches the speaker within 100 ms of the product having it: the voice's
  first chunk after its calibrated time to first audio, or the realtime server's first delta
  after the server's calibrated delays; a pause, a cut, an output replaced or the close excuse
  the wait. A realtime reply's first audio held 600 ms fails survey seed 30, a synthesis's seed
  20; clean on 12,288 seeds. None of the bugs above is in the core's last hop, so it catches none
  of them; it holds the class c8a832860 was (audio buffered before the speaker) where the core is.
- **`idle_is_cheap`** (DST rule, a cost contract): a call averages at most 200 wake-ups of its
  thread per second of virtual time (the runtime's park count; the busiest of 2,048 survey seeds
  is at 29). With a61e6faa0 reverted the room-legs seeds with background audio run at 1,000.
- **`voice_connection_reused`** (regression test, a latency contract): the production LiveKit
  Inference TTS client over a gateway that charges 200 ms of connection setup and draws its time
  to first audio from the corpus; five replies of a call, each heard within the voice's own delay
  and 50 ms. With 54c213e74 reverted reply 0 takes 417 ms against 217 ms owed; with only its
  prewarm reverted, the same.
- **`fuzz.py`** (`livekit-agents-next/scripts/fuzz.py`, run by `smoke.py`): seeded random calls
  through the real Python `VoiceSession` over `_core.FakeRoom` on the reference tests'
  virtual-time loop, with providers whose delays are drawn from `corpus/calibration.json`:
  pipeline (streaming, recognize-only over a VAD, or fallback STT; a chunked or streaming voice)
  or realtime (server turn detection or not, the default VAD or none), room options, a toolset
  that fails its setup, a plugin stream that ends in the close's cancellation, and a quarter of
  the seeds inside a console job. Per seed: nothing raises, logs a traceback or emits an error
  event; every speech finishes; the session closes once and leaves no task running; a reply asked
  of an idle agent is heard within its providers' drawn delays and 0.4 s; a realtime reply is
  asked only once earlier updates applied, and the server's turn detection is kept unless a
  detector was named; in a job the entrypoint, session end and shutdown callbacks see the job's
  context, the session is the job's primary and connected the room once, and the parent hears
  shutting down and exiting. 64 seeds take about 5 s (`smoke.py` runs them), 200 about 15 s.
- **`smoke.py`**: `decoder` (a 16 kHz WAV decodes at 16 kHz with `sample_rate=None`) and
  `llm_prewarm` (the gateway LLM sends `GET /models` as the session starts).
- **The launch gate runs the provider clients' unit tests** (`lk-agents-inference`,
  `lk-agents-abi`, about 20 s): they ran nowhere before (CI tests only the storage and A2A
  crates), so 54c213e74's own unit test could not fail anything.

### 14.2 Still outside every check

- **The binding's own tests**: the abi tests behind the Python feature (`host_tests`, 6689bcd7a's
  among them) need the binding and run in no gate. Under `--release` two abi tests fail on
  today's tree (a `should_panic` that relies on a debug assertion, and a wall-clock test); the
  gate runs them in the debug profile, where they pass.
- **Findings past the gate's window**: on today's tree a campaign over seeds 4,096 to 12,287
  finds `pause_resolves` (seed 6775), `provider_streams_closed` (7022), `speech_error_is_injected`
  (8990) and `speech_settles` (6679), none of them known findings; the gate's 5,120 seeds do not
  reach them.
- **End-to-end turn latency in the DST**: `first_audio_is_prompt` judges the last hop (audio in
  hand to the speaker); the hops before it (turn decided to request, the model's text to the
  voice) are judged by `fuzz.py`'s measured replies and by `speech_is_heard`'s 15 s.
- **What only a real room shows**: network, real provider sockets other than the realtime
  model's and the voice's, and the worker's process boundary are still exercised only by
  `latency.py` and real calls.

## 15. The other subjects

One scheduler serves every simulation (`src/sched.rs`): queued items keep their source's order,
ready processes compete with them, and BasicPOS or the arrival order picks; the voice queue adds
its due wake, frame batches and dispatch trail on top. Each subject below is a synchronous core
stepped on simulated time against fakes that answer as real peers do; its rules are functions of
the run's trace, each with counterexamples written by hand beside it; its campaign runs 4096 seeds
under both orders in the launch gate and prints its boundary (`boundary.rs`).

| Subject | Core | Fakes | Rules | Campaign |
|---|---|---|---|---|
| storage (`src/storage`) | `lk-agents-storage` plans, run commit, renewal, resolution | SQLite in memory on the simulated clock; lost acks, outages, paused and lost workers, cancels | stale owner never commits, acknowledged submissions durable, one terminal per task, uncertain commits resolve to their run, history neither lost nor duplicated | `storage_survey` |
| LLM request (`src/inference/llm.rs`) | `LlmRequest` over `Attempts` | gateway with the recorded gpt-4.1-mini chunk timing, refusals, rejections, drops, stalls, malformed and unterminated bodies, late or lost releases; a consumer with backpressure and cancels | one outcome, no replay after output, bounded backed-off retries, deadlines as the policy sets them, only the current attempt delivers, every attempt released | `inference_survey` |
| avatar session (`src/inference/avatar.rs`) | `AvatarSessionLifecycle` | idempotent gateway that applies what it received (one session per operation), answers replays from its record, loses answers, drops silent connections after 1-2 min; callers that race, leave, refuse, close | no leaked session, one session per instance, one outcome per caller, retirements with their cause, every attempt released, retirement bounded | `inference_survey` |
| process pool (`src/worker.rs`) | `PoolState` | job processes that fail to spawn, crash or hang in init, die before a launch, exit after their job, close late, with failing exit watches and reaps; duplicate dispatches, cancels, drains, the close | every admitted request settles, one disposition per assignment, retired processes stay retired, every process released and the close completes once | `worker_survey` |
| A2A door (`src/a2a`) | `lk-agents-a2a` policy resolution, `answer_pending`, `checkpoint_stop`, `settle`, `result_for` over the storage plans, under every inbox policy (`interrupt`/`queue` turns, `batch` receive and drain, `concurrent` appends, `reject`), mixed on one lane by the seed's door options, by hand as `Turn` runs them | the storage SQLite with lost acks (the connection sometimes down past the 10 s resolution wait), outages, paused and lost workers; clients on three channels that resend, continue the pending question (the door refuses a stale one), cancel at random and page `ListTasks` with tokens tried under other filters; a handler that yields at once or late | a message runs one committed turn and every answer names its task, every request answered once, with an outcome or an error the client can act on, terminal tasks never change, a cancel answers the row, a committed run's rows are what it staged (decision 39) and its history items stay once, only `interrupt` yields, one question pending, a received message is drained, a refusal only while the lane is held and with no row, listings complete and bound to their filters | `a2a_survey` (2048 seeds) |

Open A2A findings (`known_findings.tsv`, all in the runtime's `Turn` or the storage core's lease
statements): a `batch` runner's drain refused behind an older waiter, a drained batch that never
settles the waiting question, a follow-up whose named task was merged before its turn, a resent
`reject` message not deduplicated, and `concurrent` runs racing cancels and question claims.

Still to come: `SubscribeToTask` and its cursor resume are runtime code with no core decision,
outside the DST; the worker's registration and assignment state (the rest of K-2); TextSession.

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
