# Deterministic simulation tests

`lk-agents-fuzz` explores conversations, failures, and execution orders, then checks behavioral
contracts. Voice tests run the production session and runtime over controlled provider and RTC
ports. The same inputs, build, configuration, and seeds reproduce the same run.

## How it works

```text
Choose setup and inputs             search/ or a handwritten Case
    -> drive the production session drive/ over fakes/
    -> choose execution order       sched.rs and drive/queue.rs
    -> record what happened         record/
    -> check behavioral contracts   rules/ and check.rs
    -> report outcome and rerun     tests/survey.rs and drive/finish.rs
```

The generator chooses situations. Fakes implement controlled boundary behavior. Production
code makes framework decisions. Rules judge observations after the run. Changing the
generator must not change how the same case is judged.

| Tests | Real code exercised | Controlled boundary |
| --- | --- | --- |
| Voice | `VoiceSession`, its runtime loop, room/avatar executors, composed decision cores, and the voice lane's storage (`VoiceStore`, leases, renewal, fenced commit) | Provider, host, room, and media ports; the thread database connection |
| Room | Room I/O, participant streams, output, readiness, and lifecycle | The same fake RTC ports, including failures and stalls |
| Text | The AgentRun driver, TextSession and A2A turns (the runtime's `Turn`: receive, run, commit, drain) on the same production SQLite storage | The model and host callbacks (answering, panicking, never answering), the thread database connection, the clients (waiting, resending, disconnecting, cancelling, aborting) |

Storage is faked at the database connection only: the voice lane runs storage's own code over
the production SQLite backend (registry, thread files, lifecycle lock) in a directory of the
run's own, its worker tasks on the run's scheduler and its `unixepoch()` the run's clock. The
voice campaign does not exercise real provider sockets or Python callbacks.
[boundary.rs](src/boundary.rs) describes the simulation boundary and the separate conformance
suites.

## Layout

```text
src/                     the harness
  calibration.rs         measured provider timing the fakes draw their delays from
  drive/ fakes/ record/  run the production session, fake its ports, record what happened
  search/ sched.rs       choose situations and execution order
  text/                  the text driver
  rules/<domain>/        one contract per file: session, turns, speech, transcript, tools,
                         providers, realtime, avatar, storage, voice_task, text
tests/
  regressions/<domain>/  handwritten production regressions: speech, turns, tools, realtime,
                         avatar, room, session, storage
  survey.rs soak.rs      the voice campaigns (ignored by default)
  text.rs                the text driver's checker tests, regressions and campaign
  pos.rs                 scheduling-policy comparison
```

| Task | Read or edit |
| --- | --- |
| Add an action or provider delivery | [input.rs](src/input.rs), then [drive/apply/](src/drive/apply/mod.rs) |
| Change session construction | [drive/start.rs](src/drive/start.rs) and `Setup` in [drive/mod.rs](src/drive/mod.rs) |
| Change which situations are explored | [search/](src/search/mod.rs) |
| Add a behavioral contract | [rules/](src/rules/mod.rs), [check.rs](src/check.rs), and [authored.rs](src/authored.rs) |
| Understand observations and resource identity | [record/](src/record/mod.rs) and [record/views.rs](src/record/views.rs) |
| Understand scheduling | [sched.rs](src/sched.rs), then [drive/queue.rs](src/drive/queue.rs) and [drive/executor.rs](src/drive/executor.rs) |
| Understand cleanup and outcomes | [drive/finish.rs](src/drive/finish.rs) |

## Follow one production regression

Open `console_reply_plays_out::a_console_say_finishes_on_the_cli_receipt` in
[speech/playback.rs](tests/regressions/speech/playback.rs).

1. `Setup { console: true, .. }` selects production console transport with a fake CLI peer.
   [drive/start.rs](src/drive/start.rs) constructs the production `VoiceSession` through its
   ordinary interfaces; the test does not mutate private core state.
2. `Input::Say` is a developer API call. Its row in [input.rs](src/input.rs) declares payload,
   resource dependencies and production vocabulary coverage. [drive/apply/](src/drive/apply/mod.rs)
   calls `api.say`. In contrast, `Input::LlmEnd` delivers provider EOF to an already opened
   fake LLM invocation; it is not another framework API.
3. `Input::Advance` lets virtual time and runnable work progress. It does not sleep in real time.
   The fake CLI consumes production protocol bytes and supplies playback receipts.
4. `run(Case { .. })` creates one deterministic simulation, applies its inputs, releases
   remaining controlled obligations, closes, joins tasks, and checks rules. The release phase
   is observable harness assistance, not evidence that production finishes without an answer.
5. The assertion checks contract failures. For new completion regressions also assert the
   expected `Outcome::ending`, so finite exploration exhaustion cannot look like completion.

Add a regression next to the tests of its subject in `tests/regressions/<domain>/`; a new file
is declared in its domain's `mod.rs`. A campaign finding becomes a small handwritten case:
a regression never replays a generated seed (`search::explore`) or pins a schedule number,
since any change to the generator, fakes or scheduler makes the same number a different run.
Prove the case fails with the bug put back before keeping it.

## Add a contract

A file `rules/<domain>/close_finishes.rs` holds a unit struct `CloseFinishes` implementing `Rule`
(`TextRule` under `rules/text/`). `build.rs` derives the module and registry entry from the file
name. [close_finishes.rs](src/rules/session/close_finishes.rs) shows the complete workflow:

- `CloseFinishes::on_complete` reads observations and makes an expectation only when close
  was begun and the case awaited exit. It does not inspect private production state.
- `exited` is an accepted synthetic recording; `never_exited` is a rejected synthetic
  counterexample. They test whether the checker distinguishes the two timelines.
- `a_close` is a production probe: a real session receives Close and AwaitExit. It establishes
  that the rule can make an expectation against production, rather than only synthetic data.
- A production regression captures a concrete previous failure and belongs with its domain
  tests. A probe and a counterexample serve different purposes and do not replace it.

The registry checks every contract's examples, counterexamples and probes. A rule with zero
expectations is **unexercised**, even if its checker ran successfully.

## Choose a fake port

All ordinary provider invocations have ordinal identity: `Ref<Chat>::new(0)` means the first
chat opened by this run, regardless of its generated request ID. `Gone<Chat>` explicitly
selects a closed invocation for a stale-reply test. No live resource means an input requiring
one is unavailable to the generator; a failed handwritten delivery remains an observed attempt.
The input table owns generation/availability via field types (`Make`, `Referenced`, `Sampler`);
do not duplicate those rules in a generator or fake.

| Port implementation | Controls and observations | Timing, cancellation and limits |
| --- | --- | --- |
| [STT](src/fakes/stt.rs) | Case delivers the provider's reports (transcripts, its own start and end of speech, connection drops, errors) through the production speech rules (`stt::speech::Speech`); records each report with what the stream emitted, forwarded audio and opening identity | Provider VAD and `transcribes_after_speech_ends` vary by seed. Reopening creates a new invocation; late delivery targets the old one and can be rejected. No acoustic recognition |
| [VAD](src/fakes/vad.rs) | Case delivers speech/inference events; records microphone fan-out | Results are case-controlled, not inferred from audio; stream close ends delivery |
| [EOT](src/fakes/eot.rs) | Each predict call records audio/request and waits for its correlated answer | Unanswered prediction lets production timeout run; no prediction model |
| [Interruption](src/fakes/interruption.rs) | Records overlap/speaking signals and audio; case selects verdict | Production owns interruption policy; no learned overlap classifier |
| [LLM](src/fakes/llm.rs) | Records actual request IDs/tools; case sends chunks, tool calls, end or failure | Controlled stream, closed receiver rejects late delivery; no inference or HTTP |
| [TTS](src/fakes/tts.rs) | Text pushes become tagged 10 ms chunks; hold/release/fail controls and optional word alignment | Holds queue text; release drains; production cancellation closes consumption. No arbitrary synthesis |
| [Tools/hooks](src/fakes/tools.rs) | Actual call identity, arguments, host outcomes, hooks and validation fixtures | Bodies wait on case-controlled replies; production owns deadlines/approval. No Python context or external side effects |
| [Nodes](src/fakes/nodes.rs) | Records active agent/node opening and forwards to controlled providers | Exercises override plumbing and transcript forwarding, not Python execution |
| [Output/input](src/fakes/output.rs) | Records per-instance text/audio, segment receipts, microphone and recorder activity | Segment identity distinguishes replacement; case controls ordinary playback receipts. No device or real video capture |
| [Realtime](src/fakes/realtime/mod.rs) | Production inference `RealtimeMachine` handles commands, response correlation, cancellation, transcripts, history acknowledgements and reconnect replay against typed server events | [server.rs](src/fakes/realtime/server.rs) owns paced provider behavior, applied writes with delayed acknowledgements, request refusals and post-cancel output. The adapter executes typed effects and virtual deadlines; no WebSocket, serialization or native transport/retry implementation |
| [Room](src/fakes/room/mod.rs) | In-memory RTC capabilities, participants, publications, streams and faults; records where each SDK object binds to the host loop | Injected clock paces audio; production room executors own lifecycle. No SDK network behavior |
| [Avatar](src/fakes/avatar.rs) | Tagged frames/audio, buffer clear and room relay over real avatar runner wiring | Injected frame clock, generation latency and queue pressure; no model or SDK AVSynchronizer |
| [Console](src/fakes/console.rs) | Fake CLI protocol peer over in-memory bytes, paced speaker receipts | Real console transport/parser, injected timing; no OS socket or speaker |
| [Voice storage](src/fakes/storage/mod.rs) | A wrapper around each production SQLite thread connection under production `VoiceStore` and `SqliteStorage`; records each acquire/commit/abandon at the port, which lease refused a take and what the closed store left; another worker's text turns edit the lane | Seeded latency on both sides of the apply, lease takes answered after the interruption watchdog, lost answers, lost writes, outages, a database worker dying with requests queued, and ([layer](src/fakes/storage/layer.rs)) SQLite busy/locked/I/O/full refusals, `unixepoch()` clock jumps, a crash after a write applied and another worker holding or stealing the lane; no AgentDB backend |

The LLM, TTS, STT and realtime fakes draw their delays from the measured deciles in
[calibration.rs](src/calibration.rs); keep the provider and model each set was measured on.

## Realtime transcription and failure ordering

Input ASR runs independently of response generation. The fake schedules partials and finals
by the provider's item ID; it does not wait for a transcript before answering. In addition to
measured delays and seeded stalls, `RealtimeDelays` can hold a final for 20–30 seconds, finish
a later turn first, and deliver partial text long before its final:

| Control | What it exercises |
| --- | --- |
| `transcription`, `first_transcription`, `transcription_partial` | Late finals, reversed turn completions, and partial/final separation |
| `RtEvent::TranscriptionFailed`, `OldestTranscriptionFailed` | Failure before any text, after partial text, or in an older job after a newer turn completed |
| `ResponseFailed`, `ServerError`, `FatalError` | Response failure and recoverable or terminal provider errors |
| `Disconnected`, `Reconnected` | Lost connection, partial settlement and replay into a fresh server |
| `context_ack` | Applied conversation edits whose acknowledgements arrive later or are lost with the connection |
| `StorageSetup::lose_commit_answer_at` | A history commit applied once with its first answer lost; the real storage resend must recover it |
| `StorageSetup::outage` | A refused checkpoint must produce an explicit storage-error close |

The transcription regressions read the real SQLite partition back after close. They assert
stable user item identity, conversation order despite completion order, absence of duplicate
local-recognition rows, and preservation of received partials when ASR fails or reconnects.
A failed ASR job that supplied no words retains its empty model placeholder without emitting
an invented human transcript. Provider-compatible machine tests also exercise xAI cumulative
corrections and revisable completion envelopes across failure, reconnect and close.

These orderings follow the [OpenAI server-event contract](https://developers.openai.com/api/reference/resources/realtime/server-events),
[OpenAI transcription ordering guidance](https://developers.openai.com/api/docs/guides/realtime-transcription),
[xAI voice-event contract](https://docs.x.ai/developers/rest-api-reference/inference/voice), and
[Azure Voice Live event contract](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live-api-reference-2026-04-10).
OpenAI explicitly permits input ASR to finish after response events and different turns' finals
to finish out of order. xAI cumulative updates can revise earlier words, so they replace rather
than append to the prior partial.

Two bounded synthetic-audio probes on 2026-11-17, using `gpt-realtime-mini` with
`gpt-4o-mini-transcribe` and `whisper-1`, both observed the assistant's `response.done` before
the user's transcription completion. Those samples confirm the ordering case; they do not
replace the existing measured latency distribution or define a maximum ASR delay.

## Follow execution and observations

1. Search samples `Setup` (production options/ports) and inputs. Conversation actors separate
   person, developer, provider and call-ending behavior with independent random streams.
   Uniform actions supplement them with adversarial deliveries; reach is not realism.
2. `Run::apply` invokes the API or delivers to a boundary; intentional bursts need not yield.
3. `drive/queue.rs` admits external events and microphone batches; `sched.rs` selects an
   enabled source/task turn. Every source retains FIFO. Arrival mirrors production's ready
   event queue before microphone frames; POS can race the sources. Due production wakes are
   handled by the executor before queued external delivery.
4. `drive/executor.rs` hands the selected step to the real runtime. TaskBridge additionally
   controls individual injected tasks. `Gate`, slots and wakers implement that boundary;
   task cancellation and joins still belong to production task owners.
5. `drive/collect.rs` reads fake calls, returned API futures, host publications, public speech
   handles and product journal entries without granting an extra runtime turn. Rules consume
   `record/views.rs`; resource resolution uses `record/resources.rs`. Do not manufacture an
   expected production state machine in the recording.
6. `drive/finish.rs` releases case-held obligations, seeks quiet, closes, joins tasks, checks
   contracts and reports. Clock advancement, scheduler starvation refresh, finite no-progress
   limits and the real-time watchdog are different mechanisms. A finite bound alone proves
   neither deadlock nor livelock.

## Run and reproduce

From the repository root, after the normal development setup:

```bash
# Default tests, including checker examples and production regressions. Campaigns are ignored.
uv run --no-sync cargo test --manifest-path rust/Cargo.toml --release -p lk-agents-fuzz

# One regression.
uv run --no-sync cargo test --manifest-path rust/Cargo.toml --release \
  -p lk-agents-fuzz --test regressions a_console_say_finishes_on_the_cli_receipt

# One voice seed. Preserve the original budget and any explicit policy override.
VOICE_SEED=4372 VOICE_BUDGET=3000 VOICE_POLICY=TaskBridge \
  uv run --no-sync cargo test --manifest-path rust/Cargo.toml --release \
  -p lk-agents-fuzz --test survey -- --ignored --nocapture

# A small voice campaign. Omit VOICE_POLICY to use each seed's selected policy.
VOICE_SEEDS=8 VOICE_BUDGET=3000 \
  uv run --no-sync cargo test --manifest-path rust/Cargo.toml --release \
  -p lk-agents-fuzz --test survey -- --ignored --nocapture
```

`--test soak` runs longer conversations, `--test pos` compares scheduling policies and
`--test text` holds the text campaign (`TEXT_SEEDS`, `TEXT_FIRST_SEED`, `TEXT_SEED`), each
with `-- --ignored --nocapture`. [`dst.sh`](dst.sh) (`make -C rust dst`) runs the fuzz suites
and the full campaigns; it needs cargo-nextest.

`Arrival` uses the baseline source ordering, `BasicPos` samples event-source and task-pool
turns, and `TaskBridge` also samples individual bridged task turns. These are scheduling
policies, not three different generators. Neither sampling policy explores every possible
interleaving. Core transitions and their internal signal order stay synchronous.

Add `VOICE_TRACE=1` for applied inputs and `VOICE_JOURNAL=1` for journal output. `VOICE_HANG_S`
controls an independent real-time watchdog; it aborts a stuck process, real time does not
decide behavioral correctness. Replay requires the same code, configuration and random-stream
partitioning; a seed is not a permanent replay artifact.

## Diagnose a result

1. Read **ending** first: Completed, ContractViolated, Budget, Deadlock or Livelock. Check the
   evidence attached to the last three. `Outcome::ok()` only checks recorded contract failures.
2. Read the failed contracts and expectation counts; an unexercised rule proves nothing.
3. Copy the printed rerun command, preserving seed, budget and policy. The detailed report
   prints effective setup and schedule. Record repository revision and local changes too.
4. Use `VOICE_TRACE=1` to see applied inputs, then `VOICE_JOURNAL=1` for production decisions.
   Folded dispatch order detects replay differences; the recent dispatch/pending-work trail
   locates a stalled operation.
