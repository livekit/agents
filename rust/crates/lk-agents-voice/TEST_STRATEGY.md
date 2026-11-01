# Voice test strategy

How the Rust voice driver (`lk-agents-voice`) is tested, and how to use the suite. The suite is
the crate `rust/crates/lk-agents-fuzz`; what it reaches and misses, with measured numbers, is in
its `COVERAGE.md`. Product decisions still open are in section 9.

## 1. Purpose

Find unknown interactions across the driver's API, callbacks, providers, media and lifecycle,
without having to name a dangerous combination before it is explored, and judge every run against
explicit behavioural contracts. Priorities, as agreed: missing progress first, incorrect behaviour
second, ownership and lifecycle third.

Principles the suite holds to:

- **The driver is the only thing under test.** The fakes deliver what the case chose and record
  what they were given; every provider event (a transcript, a detector tick, a playout report, a
  socket drop) is an input the case sends, never something a fake derives.
- **Inputs are the actual public API payloads, and the real interface is not bypassed.** Where
  the suite still goes around it (it builds the driver with `DriverBuilder` rather than the
  runtime's `VoiceSession`), `COVERAGE.md` section 6 says so and what it costs.
- **Correctness comes from contracts**, never from the Python reference or from what the current
  implementation happens to do. One observable guarantee per rule.
- **Verdicts depend on no clock but the simulation's**, so a slow machine cannot change one.
- **Simple over clever.** A seed reproduces its run; there is no replay format, saved corpus,
  shrinker or learning across seeds.

## 2. The harness

All under `rust/crates/lk-agents-fuzz/src`:

| Piece | What it is |
|---|---|
| `input.rs` | Everything a case can send, one table: API calls, provider deliveries, room events, clock advances, awaits. Dependencies are field types (`Ref<K>` names the n-th resource of a kind the run opened, `Gone<K>` one already closed), so an input is unavailable when nothing it needs is live. |
| `drive/` | A real session with the fakes attached. `start.rs` builds it, `apply.rs` turns one input into the production command or delivery, `collect.rs` records what became true, `finish.rs` releases what is owed, closes and reports. `queue.rs` owns the driver's event queue. |
| `fakes/` | Controlled providers and outputs: STT, VAD, end of turn, interruption detector, LLM, TTS, realtime, tools, audio and text outputs. |
| `record.rs`, `record/views.rs` | What a run leaves behind, as stamped data: the driver's own journal verbatim, plus the other side of each boundary (what the case sent and whether it was refused, what the fakes produced, what left the session, what the host received). Rules read only the views. |
| `rules/` | The contracts, one per file (section 3). |
| `check.rs` | How a rule reports (`Check::expect`), the `Rule` and `Contract` types, `check_all`, and `KNOWN_FINDINGS`. |
| `authored.rs` | Hand-written recordings for the rules' examples and counterexamples. |
| `search/` | The campaign: `setup.rs` draws the session's setup, `conversation.rs` the run (section 5). |

A run is deterministic. `drive::simulate_with(schedule, ..)` gives each case its own
single-threaded Tokio runtime with paused virtual time and a scheduler RNG pinned by the case's
`schedule` (the `tokio_unstable` cfg in `.cargo/config.toml` exposes `rng_seed`).
Applying an input does not yield: only `Advance` and the awaits reach the executor, so a case
can build a burst the driver untangles in one cycle. `drive::queue::Policy` picks the next event
the driver sees: `Arrival` (the channel's order) or `Pos` (partial-order sampling over the event
sources and the driver's task pool), drawn per seed.

Every run ends the same way (`drive::finish`): release rounds answer everything left pending
(end streams and chats, answer tools and hooks, report playouts, release holds, close
utterances) until two rounds in a row have nothing to release and leave no deadline the driver
armed ahead of the clock (the recorder's periodic flush aside), then `Settle`, `Close { drain:
true }`, `AwaitExit`, and every spawned task is joined. A run still producing work after 100
rounds is reported as `session_settles`; a round that grants the next of the case's queued
floor-hold asks is working down a finite backlog and does not count. The rules then run once,
over the finished recording.

## 3. Contracts and the known-findings baseline

A rule is a file `src/rules/<id>.rs` holding a unit struct that implements `check::Rule`:

- `CONTRACT`: the id (the file name), the guarantee in one sentence, `examples` it must accept
  while making at least one expectation, `counterexamples` it must reject, and an optional
  `probe`, a whole `Case` that reaches the rule's prerequisite deterministically.
- `on_complete(recording, check)`: judges the finished run. A rule that returns before its first
  `expect` did not apply, and reports count that as unexercised, not as a pass.

`build.rs` discovers the files, so adding a rule is adding a file. The registry tests
(`src/rules/mod.rs`) fail a rule with no example, no counterexample, an example it rejects or
makes no expectation on, a counterexample it accepts, a duplicate id, an id that is not its file
name, or a failure on an empty recording. `every_probe_reaches_its_contract`
(`src/drive/finish.rs`) runs every probe and requires it to pass with at least one expectation.
A counterexample validates the checker, not the runtime.

`check::KNOWN_FINDINGS` lists the contracts with open findings. A finding of any other contract,
anywhere in the gate, fails it, so a new kind of defect cannot hide among the known ones. A
contract whose findings are all fixed comes off the list; a registry test fails if the list names
something that is not a contract (`session_settles`, the release phase's own verdict, excepted).

## 4. Progress, hangs and close

Missing progress is found without real time:

| Contract | What it catches |
|---|---|
| `virtual_time_advances` | A livelock: the session works without bound at one virtual instant. The run counts steps at one instant, in the journal and in its own loop, and stops the session past `NO_PROGRESS_STEPS` (10,000, twenty times the busiest instant measured). The run ends there and the finding is reported, rather than the campaign hanging. |
| `every_task_finishes` | A stuck task: every task the session spawned (the harness's `Spawner` keeps each join and the line that spawned it) must finish, without panicking, once the session closed. A task still pending when its join's virtual budget (`BUDGET_MS`, 30 s) runs out is aborted and reported as stuck, with its spawn site. |
| `close_finishes` | A close that never ends. Once a close is begun, `AwaitExit` waits however long it takes in virtual time; a session not exited a virtual day later sits with nothing that could wake it. |
| `nothing_left_live` | Turn-owned work left open after the driver exits: a floor hold, tool, hook call, filler, segment, utterance, prediction, realtime turn or say stream. |
| `every_resource_released` | A leaked registration: once the driver exited, the session's resource table (`lk_agents_abi::resources`) must be empty. |

Related ones: `driver_exits_only_on_close` (no exit without a close, never by panic),
`speech_settles`, `api_calls_return`, `provider_streams_closed`. Every other wait the harness
makes is bounded in virtual time (`BUDGET_MS`), and every loop in rounds (release, join).

What no step count can see, a product task that never yields and blocks the run's only thread,
is the watchdog's: a thread beside each simulation aborts the process after `VOICE_HANG_S` real
seconds (600 by default), printing the seed and the line that reruns it with `VOICE_TRACE=1`.
That is a crash, not a verdict.

The soak (`tests/soak.rs`, eight calls with a budget of 12,000 inputs each by default) adds what only long calls
show: per resource kind, the peak live count by quarter of the call and what is left live after
the release phase. A call that does not reach its close, a residue of anything that does not
live with the session, or a finding outside the known ones fails it.

## 5. Search-space inventory

Every driver boundary enum carries `#[derive(Vocabulary)]` (`lk-agents-abi/src/utils/
vocabulary.rs`), and the input table must declare each variant covered or skipped with a reason
(`assert_command_is_covered` in `input.rs`): a new command the suite does not handle fails to
compile. The journal's step names (`lk_agents_voice::trace::step_names`) are the reach metric:
the survey reports which variants a campaign reached and which it never did.

One generator, `search::explore(setup, seed, budget)`. Everything in a run is drawn from its seed
alone: the setup (`search::sample_setup`, every switch a coin), the conversation, the providers'
behaviour, the queue policy and the scheduler. Four actors each act only on what they observed: a
person who speaks, pauses, barges in, backchannels and waits their own patience; the developer's
program, calling the API between and inside turns and answering hooks and tools; the end of the
call, its own dial; and providers that answer and fail as real ones do. Before each conversation
move a run may also make uniform moves (any legal row except one that ends the call), at a rate
drawn per seed over seven decades (`search::randomness`), so the search spans a pure conversation
and a pure uniform sampler. Uniform moves are labelled in `Case::random`.

The search's yardstick is reach (journal variants, distinct edges between consecutive steps,
input x phase cells) and distinct findings; `COVERAGE.md` sections 7 to 10 hold the numbers and
the biases the search still has.

## 6. The launch gate

One command, from the repository root:

```bash
rust/crates/lk-agents-fuzz/launch-gate.sh
```

It runs, in order, failing at the first problem:

1. `cargo check -q --workspace --all-targets`
2. `cargo test -q -p lk-agents-voice`: the driver's unit tests.
3. `cargo test -q --release -p lk-agents-fuzz`: the registry and probe tests, the search's own
   invariants, `tests/determinism.rs`, and every regression test.
4. The survey on seeds 0 to 1023 (`tests/survey.rs`).
5. A campaign on the next `VOICE_GATE_SEEDS` seeds (4096 by default).
6. The soak.

Each campaign step must run its one ignored test and pass it; the survey and the soak fail on
any finding of a contract outside `KNOWN_FINDINGS`. `VOICE_HANG_S` bounds a blocked run.

Campaigns by hand (from `rust/`):

```bash
cargo test --release -p lk-agents-fuzz --test survey -- --ignored --nocapture
cargo test --release -p lk-agents-fuzz --test soak -- --ignored --nocapture
```

Survey knobs: `VOICE_SEEDS` (32), `VOICE_FIRST_SEED` (0), `VOICE_BUDGET` (1500 inputs),
`VOICE_CELLS=1` (the input x phase table), `VOICE_FAMILY=<Family>` (one journal family variant by
variant). Soak knobs: `VOICE_SOAK_SEEDS` (8), `VOICE_SOAK_BUDGET` (12,000).

## 7. From a finding to a regression test

1. **Reproduce it from its seed.** The survey reports, per failing contract, an example seed and
   the line that reruns it alone:

   ```bash
   cd rust
   VOICE_SEED=4372 cargo test -q --release -p lk-agents-fuzz --test survey -- --ignored --nocapture
   ```

   That prints the run's setup, randomness, queue policy, every failure in full and the journal
   variants it reached. `VOICE_TRACE=1` narrates every input as it is applied. A seed reproduces
   its run on the same code only; `tests/determinism.rs` checks that one seed explores one case
   and that a case leaves the same recording twice.
2. **Understand it** from the trace and the failure's subject (a speech by id, a resource by
   kind and index, or the session).
3. **Write a regression test** in `rust/crates/lk-agents-fuzz/src/*_tests.rs` (the suite for that behavior): a hand-written `Case`
   (or a `Run` driven input by input) that reaches the bug through the fewest inputs, fails
   before the fix and passes after it, with a `// Catches:` comment naming the defect and the
   seed it came from. It holds whatever the search later changes. See `hold_at_close.rs` or
   `close_outlasts_await.rs`.
4. **Fix it**, and once a contract has no open findings left, take it off `KNOWN_FINDINGS`.

A new contract follows the same path in the other direction: write the rule with its examples,
counterexamples and probe, run the survey, and triage what it finds before deciding whether its
findings are known.

## 8. What is not covered

- **No shrinker.** A finding is reduced by hand while writing its regression test. An
  obligation-preserving shrinker over a seed's inputs is a possible later addition.
- **No cross-seed learning or corpus.** Each seed is uniform over its setup; keeping cases or
  weighting by earlier seeds would make a seed depend on the seeds before it.
- **Internal scheduling is partly controlled.** The queue policy orders events into the driver,
  and Tokio's scheduler RNG is seeded per case, but tasks run on one thread in Tokio's order and
  multi-thread interleavings are not explored.
- **The public API and room I/O are not driven** (`COVERAGE.md` section 6), and the Python
  pipeline, bindings, real providers and other crates are outside this suite (section 11 there).
- **No finite campaign shows an interaction impossible**, and a line or variant reached is not a
  guarantee checked.
- **A stuck agent that stays silent is not caught** until per-operation progress bounds are
  decided (section 9).

## 9. Decisions still needed

Each blocks one family of checkers. The harness can observe the facts each needs and the
generator reaches the states; what is missing is the product decision, which must state the
allowed outcomes, the progress and deadline policy, the cancellation behaviour, and one accepted
and one rejected example. Nobody invents one or copies the reference as approval.

| Decision | The question | Options seen so far |
|---|---|---|
| Meaningful progress per operation | Which observations reset an operation's inactivity timer, and what is the limit? Queue time counts from the request. | Per leg: a chat delta, a synthesized chunk, a sink push. Whether a keep-alive, a retry or a state transition counts is open |
| Expiry handling | When an operation passes its inactivity limit, what does the driver owe? | (a) a retry within the leg's budget; (b) fail the speech with a provider error; (c) close the session |
| Turn-authorization permission | Per turn mode, the independent predicate that says when a reply may be authorized. Without it `playback_requires_authorization` checks only that playback follows authorization. | One predicate per mode from the configured turn policy and the case's own inputs, never from the driver's `Authorized` step |
| Event-only waits | `wait_for_interrupted` stays pending after a speech completes normally. Is that waiter owed an answer? | (a) pending forever, the binding must not leak it; (b) resolves "not interrupted" at speech end; (c) cancelled at speech end |
| Invalid provider events | Which provider sequences are forbidden rather than merely late, and what is owed for each? | Late delivery after cancellation is normal coverage. Duplicate finals, a final before a start, a chunk after end are candidates for a robustness campaign |
| Competing handoffs and tool continuation | When two parallel tool calls both hand off, which wins? When may a tool result start a follow-up, for server-driven and client-driven continuation? | The allowed winners and refusals must be named first |
| Pending-tool history | A new request arrives while a tool runs: does it wait, omit the call, or carry a placeholder? | All three are checkable once chosen |
| Output shutdown | Per output, which operations must complete on close, under what deadline, and when forced cleanup is allowed. | `provider_streams_closed` covers provider streams only; output close is unchecked |
| Acoustic echo | Whether echo discrimination is in scope, and at which layer. | An adapter-layer test with labelled audio, or a declared gap |

The first three matter most: progress and expiry turn "a speech never finished" into a product
contract, and turn-authorization permission completes the only speculative-playback contract.
