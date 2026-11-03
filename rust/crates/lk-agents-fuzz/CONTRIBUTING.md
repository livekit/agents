# Contributing a voice or room simulation

Start with [README.md](README.md) for the testing boundary and normal commands. This guide
follows one test, then explains the mechanisms needed when extending it. Framework policy
always comes from production code. Fakes supply controlled port behavior.

## Follow one production regression

Open `console_reply_plays_out::a_console_say_finishes_on_the_cli_receipt` in
[speech_playback_tests.rs](src/speech_playback_tests.rs).

1. `Setup { console: true, .. }` selects production console transport with a fake CLI peer.
   [drive/start.rs](src/drive/start.rs) constructs the production `VoiceSession` through its
   ordinary interfaces; the test does not mutate private core state.
2. `Input::Say` is a developer API call. Its row in [input.rs](src/input.rs) declares payload,
   resource dependencies and production vocabulary coverage. [drive/apply/mod.rs](src/drive/apply/mod.rs)
   calls `api.say`. In contrast, `Input::LlmEnd` delivers provider EOF to an already opened
   fake LLM invocation; it is not another framework API.
3. `Input::Advance` lets virtual time and runnable work progress. It does not sleep in real time.
   The fake CLI consumes production protocol bytes and supplies playback receipts. The second
   speech catches accidental dependence on a receipt from the first speech.
4. `run(Case { .. })` creates one deterministic simulation, applies its inputs, releases
   remaining controlled obligations, closes, joins tasks, and checks rules. The release phase
   is observable harness assistance, not evidence that production finishes without an answer.
5. The assertion checks contract failures. For new completion regressions also assert the
   expected `Outcome::ending`, so finite exploration exhaustion cannot look like completion.

Run this test from the repository root:

```bash
uv run --no-sync cargo test --manifest-path rust/Cargo.toml --release \
  -p lk-agents-fuzz --test regressions a_console_say_finishes_on_the_cli_receipt
```

Add a regression to the existing subject's sibling `*_tests.rs`. New behavior modules must
be included by [regressions_tests.rs](src/regressions_tests.rs); explicit independent Cargo
processes belong in [Cargo.toml](Cargo.toml). The build-registration guard rejects forgotten
modules. Prefer a small explicit case over reproducing an entire generated conversation.

## Add a guarantee without confusing checker tests with product tests

[close_finishes.rs](src/rules/close_finishes.rs) shows the complete workflow:

- `CloseFinishes::on_complete` reads observations and makes an expectation only when close
  was begun and the case awaited exit. It does not inspect private production state.
- `exited` is an accepted synthetic recording; `never_exited` is a rejected synthetic
  counterexample. They test whether the checker distinguishes the two timelines.
- `a_close` is a production probe: a real session receives Close and AwaitExit. It establishes
  that the rule can make an expectation against production, rather than only synthetic data.
- A production regression captures a concrete previous failure and belongs with its domain
  tests. A probe and a counterexample serve different purposes and do not replace it.

A file `rules/close_finishes.rs` contains unit struct `CloseFinishes` implementing `Rule`.
`build.rs` derives the module and registry entry from the filename. `Contract` holds the
identifier, guarantee, examples, counterexamples and optional probe; `Registered` lets the
registry invoke the concrete rule; `Check` accumulates expectations and failures. Existing
rules may use a shared production probe; inspect the registry/probe checks before adding one.
A rule with zero expectations is **unexercised**, even if its checker ran successfully.

## Find rules by domain

The directory is the live index; these representative entry points group the responsibilities
without duplicating a count that drifts:

| Domain | Start here |
| --- | --- |
| Progress, close and tasks | [close_finishes](src/rules/close_finishes.rs), [virtual_time_advances](src/rules/virtual_time_advances.rs), [every_task_finishes](src/rules/every_task_finishes.rs), [api_calls_return](src/rules/api_calls_return.rs) |
| Turns, endpointing and interruption | [eot_judged_by_the_detector](src/rules/eot_judged_by_the_detector.rs), [turn_hook_is_awaited](src/rules/turn_hook_is_awaited.rs), [interruption_has_cause](src/rules/interruption_has_cause.rs), [refused_interrupt_is_no_interruption](src/rules/refused_interrupt_is_no_interruption.rs) |
| Speech, playback and transcript | [speech_settles](src/rules/speech_settles.rs), [audio_conservation](src/rules/audio_conservation.rs), [playback_requires_authorization](src/rules/playback_requires_authorization.rs), [transcript_follows_audio](src/rules/transcript_follows_audio.rs) |
| Tools and approval | [tool_lifecycle](src/rules/tool_lifecycle.rs), [tool_output_follows_outcome](src/rules/tool_output_follows_outcome.rs), [approval_runs_on_the_answer](src/rules/approval_runs_on_the_answer.rs) |
| Options and ownership | [option_comes_from_its_layer](src/rules/option_comes_from_its_layer.rs), [every_resource_released](src/rules/every_resource_released.rs), [output_replacement_is_clean](src/rules/output_replacement_is_clean.rs) |
| Room, realtime and avatar | [realtime_server_accepts_requests](src/rules/realtime_server_accepts_requests.rs), [realtime_server_holds_what_was_heard](src/rules/realtime_server_holds_what_was_heard.rs), [avatar_av_in_sync](src/rules/avatar_av_in_sync.rs), [room tests](src/room_io_tests.rs) |
| VoiceTask | [voice_task_restores_its_parent](src/rules/voice_task_restores_its_parent.rs), [voice_task_returns_its_value](src/rules/voice_task_returns_its_value.rs), [voice_task_skips_parent_hooks](src/rules/voice_task_skips_parent_hooks.rs) |

## Choose a fake port

All ordinary provider invocations have ordinal identity: `Ref<Chat>::new(0)` means the first
chat opened by this run, regardless of its generated request ID. `Gone<Chat>` explicitly
selects a closed invocation for a stale-reply test. No live resource means an input requiring
one is unavailable to the generator; a failed handwritten delivery remains an observed attempt.
The input table owns generation/availability via field types (`Make`, `Referenced`, `Sampler`);
do not duplicate those rules in a generator or fake.

| Port implementation | Controls and observations | Timing, cancellation and limits |
| --- | --- | --- |
| [STT](src/fakes/stt.rs) | Case delivers recognition events, end and errors; records forwarded audio and opening identity | Reopening creates a new invocation; late delivery targets the old one and can be rejected. No acoustic recognition |
| [VAD](src/fakes/vad.rs) | Case delivers speech/inference events; records microphone fan-out | Results are case-controlled, not inferred from audio; stream close ends delivery |
| [EOT](src/fakes/eot.rs) | Each predict call records audio/request and waits for its correlated answer | Unanswered prediction lets production timeout run; no prediction model |
| [Interruption](src/fakes/interruption.rs) | Records overlap/speaking signals and audio; case selects verdict | Production owns interruption policy; no learned overlap classifier |
| [LLM](src/fakes/llm.rs) | Records actual request IDs/tools; case sends chunks, tool calls, end or failure | Controlled stream, closed receiver rejects late delivery; no inference or HTTP |
| [TTS](src/fakes/tts.rs) | Text pushes become tagged 10 ms chunks; hold/release/fail controls and optional word alignment | Holds queue text; release drains; production cancellation closes consumption. No arbitrary synthesis |
| [Tools/hooks](src/fakes/tools.rs) | Actual call identity, arguments, host outcomes, hooks and validation fixtures | Bodies wait on case-controlled replies; production owns deadlines/approval. No Python context or external side effects |
| [Nodes](src/fakes/nodes.rs) | Records active agent/node opening and forwards to controlled providers | Exercises override plumbing and transcript forwarding, not Python execution |
| [Output/input](src/fakes/output.rs) | Records per-instance text/audio, segment receipts, microphone and recorder activity | Segment identity distinguishes replacement; case controls ordinary playback receipts. No device or real video capture |
| [Realtime](src/fakes/realtime/mod.rs) | Controlled server events, requests, conversation items, generation and connection faults | [server.rs](src/fakes/realtime/server.rs) owns paced provider behavior; production owns session policy. No native transport/retry implementation |
| [Room](src/fakes/room/mod.rs) | In-memory RTC capabilities, participants, publications, streams and faults; records where each SDK object binds to the host loop | Injected clock paces audio; production room executors own lifecycle. No SDK network behavior |
| [Avatar](src/fakes/avatar.rs) | Tagged frames/audio, buffer clear and room relay over real avatar runner wiring | Injected frame clock, generation latency and queue pressure; no model or SDK AVSynchronizer |
| [Console](src/fakes/console.rs) | Fake CLI protocol peer over in-memory bytes, paced speaker receipts | Real console transport/parser, injected timing; no OS socket or speaker |
| [Voice storage](src/fakes/storage.rs) | Voice lane acquire/commit/abandon/loss and seeded concurrent history edits | Immediate port responses unless configured lease loss; no database/fencing/renewal implementation |

For larger models, follow room capability implementations from `fakes/room/mod.rs`, realtime
session wiring into `realtime/server.rs`, and avatar generator/synchronizer from `avatar.rs`.
[boundary.rs](src/boundary.rs) maps these fakes to owning transport/component conformance suites.
[calibration.json](calibration.json) is reference measurement data. Executable timing arrays
are maintained manually in `realtime/server.rs` and conversation actors; editing JSON alone
changes no campaign. Carry measurement identity when changing those constants.

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

Keep `Run`, `Recording` and input exports available to explicit integration targets. Splitting
one of these files is justified by a distinct responsibility, not by introducing another
facade with the same operations. Change scheduling only with exact replay/source FIFO tests,
matched seed/policy comparisons, cancellation/due-wake tests and known-bug search evidence.

## Diagnose and reproduce a result

1. Read **ending** first: Completed, ContractViolated, Budget, Deadlock or Livelock. Check the
   evidence attached to the last three. `Outcome::ok()` only checks recorded contract failures.
2. Read NEW FINDING versus KNOWN FINDING and expectation counts. A tolerated finding is still
   a violation; an unexercised rule proves nothing. Remove a fixed baseline entry only with
   its regression and the owning gate.
3. Copy the printed rerun command, preserving seed, budget and policy. The detailed report
   prints effective setup and schedule. Record repository revision and local changes too.
4. Use `VOICE_TRACE=1` to see applied inputs, then `VOICE_JOURNAL=1` for production decisions.
   Folded dispatch order detects replay differences; the recent dispatch/pending-work trail
   locates a stalled operation. None independently proves every ordering was explored.

| Command/target | Purpose |
| --- | --- |
| README default command | Checker tests, probes and explicit production regressions |
| `--test regressions <name>` | One concrete production behavior |
| `--test survey -- --ignored --nocapture` | Seeded reach/findings; VOICE_SEED/VOICE_SEEDS/VOICE_FIRST_SEED/VOICE_BUDGET/VOICE_POLICY configure the window |
| `--test soak -- --ignored --nocapture` | Longer conversations; inspect its own environment controls |
| `--test pos -- --ignored --nocapture` | Scheduling-policy comparison, rather than a replacement generator |
| `launch-gate.sh` | Workspace verification: architecture and non-fuzz gates as well as simulations |
| `bench/known_bugs.sh` | Reverses fixes only in a copied tree and compares known-bug discovery |

Replay requires the same code, configuration and random-stream partitioning. Seeds do not
survive arbitrary refactors. Historical COVERAGE/benchmark tables must keep revision, seed
window, policy, budget and boundary attached; derive current inventories from source rather
than presenting historical counts as today's capabilities.
