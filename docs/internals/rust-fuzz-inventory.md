# Rust fuzz crate audit and improvement inventory

The fuzz crate has a clear testing model: generate situations, run real framework code over
controlled ports, record observations, and check contracts. Understanding that model should
take a minute. Understanding a new regression should require only its domain behavior and the
small part of the harness it uses.

The main obstacle is the amount of harness detail a reader encounters before that model is
clear. Scheduling has several
layers; resource references and recordings have their own vocabulary; documentation still
contains descriptions of earlier layouts and measurements.

This audit covers all top-level module responsibilities, the voice harness and its submodules,
room tests, rule infrastructure, diagnostics, documentation, and operational tools. It identifies crate-level improvements rather than auditing every contract's semantic
correctness. Public behavior remains owned by the feature documents. See also the
[Rust transition inventory](rust-transition-inventory.md) and
[Rust core testing contracts](rust-core.md#deterministic-simulation-testing).

## Working-tree status after the plan audit

The voice/room scope reduction, README, test-module inclusion guard, reproduction command
policy/budget fields, watchdog schedule identity, and deterministic-spawner teardown fix are
present. The shared-policy implementation checkpoint records 25 fuzz-library/oracle tests
and 208 deterministic regressions passed. Those results verify the tested behavior, not every
navigation/reporting recommendation or future scheduler redesign in this inventory.

Historical measurements below remain historical. Rows R08/P02/P03 now track maintaining and
verifying implemented guards/reporting rather than diagnosing bugs still present. A worked
contributor regression walkthrough, complete local fake contracts, and remaining navigation/
reporting improvements are still work to do.

## Size and structure before scope reduction

Working-tree snapshot on 2026-11-01, before adding documentation: 138 Rust source files under
`src/`, 44,379 lines, and 64 voice rule modules. Counts include tests and exclude `build.rs`,
generated build output, shell scripts, and data files. Size explains the need for navigation;
it does not by itself justify splitting files or removing tests.

| Area | Rust files | Lines | Responsibility |
| --- | ---: | ---: | --- |
| Root modules | 7 | 2,920 | Library entry point, inputs, checks, authored records, boundaries, scheduler, TextSession |
| `drive/` | 8 | 5,154 | Voice setup, API application, queue control, observations, execution, and cleanup |
| `fakes/` | 21 | 8,799 | Controlled provider, RTC, host, output, and storage ports |
| `record/` | 3 | 2,485 | Observations, resource accounting, and rule-facing views |
| `rules/` | 65 | 10,205 | 64 voice guarantees and their registry |
| `search/` | 7 | 2,536 | Setup sampling, conversation actors, and uniform actions |
| `a2a/` | 4 | 1,470 | Requests, handler/database faults, and A2A rules |
| Root test modules | 23 | 10,810 | Regression groups, campaigns, scheduling comparisons, and surveys |

The largest navigation units include `search/conversation/mod.rs`, `record/views.rs`,
`drive/apply/mod.rs`, `record/mod.rs`, and `fakes/room/mod.rs`. Keep related logic together;
extract by domain responsibility when a module becomes hard to navigate. Splitting at a line
threshold would add files without necessarily reducing the concepts a reader needs.

## The explanation a contributor should see first

```text
Setup and actions -> real framework over fake ports -> observations -> contract results
                          ^
              deterministic clock and scheduling
```

The [crate README](../../rust/crates/lk-agents-fuzz/README.md) provides this explanation,
commands, and a task map. Five concepts should suffice for an ordinary voice test:

| Concept | Meaning | Existing names to retain |
| --- | --- | --- |
| Case | Configuration, actions, scheduling seed, and policy | `Case`, `Setup`, `Input` |
| Run | One live simulation of that case | `Run` |
| Fake | Controlled implementation of a production port | STT, TTS, LLM, room, tool, output |
| Recording | Observed inputs, public outcomes, journal, and boundary activity | `Recording` |
| Rule | A guarantee checked against observations | `Rule`, `Contract`, `Check` where their roles differ |

Scheduling and recording internals should become relevant when editing those mechanisms.
Do not require `Gate`, `Slot`, `Observed`, `Tag`, or registry generation knowledge
to understand a straightforward `say`, interruption, tool, or close regression.

## Voice and room scope

The authoritative campaign descriptions are in
[boundary.rs](../../rust/crates/lk-agents-fuzz/src/boundary.rs).

| Tests | Exercises | Does not establish |
| --- | --- | --- |
| Voice | Production VoiceSession and runtime loop, composed cores, room/avatar executors over controlled ports | Correct Python context handling, actual provider transports, or a real LiveKit room |
| Room | Production room I/O over the same fake RTC ports, with participant/output lifecycle failures and stalls | Real SDK network behavior |

A2A and standalone TextSession simulations have been removed, along with their campaign
entry points, direct storage/protocol dependencies, cross-domain rule registration, and
unused scheduler timeline support. Their component and conformance tests remain in the owning
crates. Removing these campaigns also removes their randomized schedule coverage; the remaining
component tests are not claimed to provide equivalent exploration.

The DST should remain voice and room simulations over fakes. Keep the shared clock, scheduling,
observations, and contracts needed by those tests. Avoid adding separate backend worlds or a
general simulation framework. Shared agent execution and fake tools remain in scope when they
participate in a voice interaction.

## Simplification priorities

1. Keep one voice harness and room tests using the same fake RTC ports.
2. Present one normal way to run, reproduce, and inspect a case; keep policy experiments behind it.
3. Keep domain names such as speech, turn, tool, participant, and output visible in tests.
4. Keep generators, fakes, observations, and contract checks separate by responsibility.
5. Remove unused cross-domain abstractions instead of replacing them with new wrappers.

## Entry points and documentation inventory

Disposition: **recommended** means improve the indicated contract or navigation;
**investigate** means establish the benefit and behavioral equivalence before restructuring.
Items describe proposals unless explicitly marked delivered.

| ID | Finding or opportunity | Action | Disposition |
| --- | --- | --- | --- |
| D01 | No short crate-level introduction previously existed | Add the one-minute model, campaign boundaries, commands, and task map | Delivered in `README.md` |
| D02 | Scope and entry points need to agree | Keep voice/room scope explicit in the library, README, gate, and boundary report | Delivered |
| D03 | Module comments refer to `record.rs`, `drive.rs`, `search.rs`, `conversation.rs`, and `tests/survey.rs` after layout changes | Correct current navigation references without rewriting historical measurements as current facts | Recommended |
| D04 | Cargo comments mention a supervised campaign `run.py` | Replace obsolete launcher guidance with current test targets and scripts | Recommended |
| D05 | Historical measurements need separation from current capability descriptions | Historical notice is present; audit remaining capability/navigation claims and keep dates/revisions on measurements | Partially delivered |
| D06 | Strategy documentation and the crate entry point overlap | Keep the README short and link to the detailed voice strategy | Recommended |
| D07 | Understanding one task requires reading multiple broad documents | Maintain a task-to-file/test map and one worked production regression walkthrough | Partially delivered; walkthrough remains |
| D08 | Source counts and rule counts can drift | Derive current inventories when reporting them, or label snapshots explicitly | Recommended |

Concrete stale descriptions include the coverage report's 37-rule snapshot, fixed Tokio task
order description, and older list of unmodeled features now represented by regression modules
or fakes. Preserve their historical measurement scope; they must not define today's coverage.

## Module and responsibility inventory

| ID | Area | Improvement | Constraint |
| --- | --- | --- | --- |
| M01 | `input.rs` | Explain the input table with one host-call row and one provider-delivery row, including generated behavior | Keep one declaration and compile-time production-vocabulary coverage |
| M02 | `Ref<K>` and `Gone<K>` | Show ordinal identity, live references, late replies, and unavailable inputs with concrete examples | Preserve reproducible resource resolution and explicit stale-input testing |
| M03 | Input `Make`, `Referenced`, and `Sampler` | Keep generation mechanics behind the table for ordinary contributors; explain extension points locally | Avoid duplicated availability logic |
| M04 | `drive/mod.rs` | Separate the public harness API and domain setup from scheduler/watchdog implementation in navigation | Keep one run lifecycle; do not introduce another harness facade |
| M05 | `drive/start.rs` | Group construction by familiar session options and ports | Construction must use the production interface, not mutate private core state |
| M06 | `drive/apply/` | Group action handling by domain and link each input to its API call or boundary delivery | Applying ordinary inputs must preserve burst behavior and intentional yield points |
| M07 | `drive/collect.rs` | State which observations come from the product journal, fake calls, and returned host results | Do not invent ownership events or derive expected behavior from private state |
| M08 | `drive/finish.rs` | Make release, quiet detection, close, task joins, checks, and reporting separate visible phases | Automatic release can affect liveness; preserve its progress assumptions and record its actions |
| M09 | `record/mod.rs` | Separate observation vocabulary from resource bookkeeping by responsibility | Avoid a parallel expected-state implementation of production policy |
| M10 | `record/views.rs` | Group views by speech, API, tools, outputs, inference, and lifecycle; show preferred views per rule | Preserve identity, timing, errors, and ordering needed by contracts |
| M11 | `record/resources.rs` | Explain resource identity and who records opening/closing, including non-journaled resources | Retain one accounting source for each resource kind |
| M12 | `authored.rs` | Clearly mark synthetic recordings as checker fixtures | A hand-authored expected record does not establish production behavior |
| M13 | `search/setup.rs` | Map sampled switches to session options and show fixed/default options separately | A sampled setup is not evidence that its behavior was exercised |
| M14 | `search/conversation/` | Describe person, developer, provider, and call-ending responsibilities before their controls | Preserve coherent causality and independent randomness streams |
| M15 | `search/uniform.rs` | Explain when unrestricted actions supplement conversation actors | Label these actions in results; distinguish legal adversarial events from realistic provider behavior |
| M16 | `fakes/` | Give every fake a short port contract: accepted controls, recorded calls, timing, cancellation, and unsupported behavior | Keep framework policy in production; preserve meaningful boundary lifecycle behavior |
| M17 | Realtime, room, avatar, and console fakes | Provide local maps for these larger protocol models and identify corresponding conformance suites | Avoid growing alternative session implementations inside fakes |
| M18 | `calibration.json` and hardcoded timings | Make the reference-only status explicit and document how measured values reach executable fake constants | Do not imply changing the JSON changes a campaign |
| M19 | Backend A2A simulation | Remove the simulation, its faults/rules/campaign, and direct storage dependency | Delivered; owning component tests remain |
| M20 | Standalone TextSession simulation | Remove the world, rules, checker fixtures, and campaign | Delivered; owning core/runtime tests remain |
| M21 | Public module exports | Audit which implementation details need cross-target visibility | Keep integration-test access practical; avoid cosmetic wrappers or a large generic framework |

Keep the existing names `Input`, `Case`, `Run`, `Recording`, and provider names where they
already communicate the responsibility. Expand abbreviations such as `sched` or `Src` only
where ambiguity matters; wholesale renaming would create churn without explaining execution.

## Scheduling inventory

There are three different responsibilities: the shared selection algorithm in
[sched.rs](../../rust/crates/lk-agents-fuzz/src/sched.rs), voice queue and task-bridge admission
in [drive/queue.rs](../../rust/crates/lk-agents-fuzz/src/drive/queue.rs), and control of the real
runtime loop in [drive/executor.rs](../../rust/crates/lk-agents-fuzz/src/drive/executor.rs).

| ID | Opportunity | Action and constraint |
| --- | --- | --- |
| S01 | Readers encounter several scheduler-related types without their relationship | Document selection algorithm, event admission, task polling, clock advance, and production loop handover in one sequence |
| S02 | `Sampling` and voice `Policy` expose different levels | Explain that Arrival/BasicPos are selection algorithms, while TaskBridge also changes individual task participation; do not merge enums solely for naming consistency |
| S03 | Queue comments say microphone frames precede events in arrival order, while production `Runtime::serve` lists event receive before frame receive in its biased select | Reconcile the intended baseline with actual runtime ordering and add a simultaneous-ready test before claiming production-order equivalence |
| S04 | Source causality and due-wake precedence are easy to miss | Put these invariants beside the entry API and retain independently known legal-order tests |
| S05 | Bridged task scheduling requires knowledge of `Gate`, slots, wakers, and task-pool turns | Keep these internals behind a documented task scheduling boundary; explain cancellation and completion ownership |
| S06 | Clock advancement, starvation bounds, and no-progress bounds look like one timeout mechanism | Explain each separately; finite bounds alone must not become proofs of deadlock or livelock |
| S07 | Changes to grouping or polling can change explored schedules | Compare identical seeds and policies, exact replay, reach, and known-bug detection before and after changes |

Do not replace the injected scheduler with ordinary Tokio execution for convenience. The real
runtime executors must remain under deterministic control. Internal synchronous core signals
retain their defined order; the scheduler samples external delivery and task turns.

## Contracts and test organization inventory

The implementations are in [check.rs](../../rust/crates/lk-agents-fuzz/src/check.rs),
[rules/mod.rs](../../rust/crates/lk-agents-fuzz/src/rules/mod.rs), and the
[rule discovery build script](../../rust/crates/lk-agents-fuzz/build.rs). Explicit test targets
are declared in [Cargo.toml](../../rust/crates/lk-agents-fuzz/Cargo.toml).

| ID | Opportunity | Action and constraint |
| --- | --- | --- |
| R01 | `Rule`, `Contract`, `Registered`, and `Check` require separate reading | Explain executable checker, metadata/examples/probe, generated registry entry, and expectation accumulator beside their definitions |
| R02 | Rule discovery is implicit build behavior | Document filename-to-type naming and generated registration; keep the one-file addition workflow |
| R03 | Checker examples can be mistaken for production tests | Keep accepted example, rejected counterexample, production probe, and production regression visibly distinct |
| R04 | A rule that never applies can look green | Show expectation counts and unexercised contracts prominently; preserve registry and probe checks |
| R05 | 64 rule modules are hard to browse by filename alone | Generate or maintain a domain index: progress, turns, speech/media, tools, options, ownership, RTC/avatar, and VoiceTask |
| R06 | Cross-domain rule registration adds complexity | Remove backend rule registries from the voice registry | Delivered |
| R07 | `autotests = false` and explicit targets hide how sibling test files execute | Document that `regressions_tests.rs` includes behavior modules and that campaigns have their own Cargo targets |
| R08 | Test-module registration needs enforcement | Inclusion guard is implemented in `build.rs` / `build_registration.rs`; retain its counterexamples and explicit process-isolation model |
| R09 | Known findings can be confused with passing behavior | Show new failures, tolerated known findings, and fixed findings separately; preserve the shrinking baseline and regression links |
| R10 | Search reach can be confused with correctness coverage | Separate variants, adjacent-step pairs, input/phase cells, resource opens, and actual contract expectations in reports |

## Reproduction and operational inventory

Reproduction commands and reporting live in
[survey_tests.rs](../../rust/crates/lk-agents-fuzz/src/survey_tests.rs); runtime/watchdog setup
is in [drive/mod.rs](../../rust/crates/lk-agents-fuzz/src/drive/mod.rs), and completion evidence
is in [drive/finish.rs](../../rust/crates/lk-agents-fuzz/src/drive/finish.rs).

| ID | Opportunity | Action and constraint |
| --- | --- | --- |
| P01 | Many environment variables encode a run | Print one effective configuration and complete rerun command, including seed, schedule, policy override, budget, and relevant campaign options |
| P02 | Rerun command must preserve effective seed, budget, and policy | Implemented in `survey_tests.rs`, with reproduction assertions; retain those checks under future reporting changes |
| P03 | Watchdog output must distinguish schedule and campaign identity | Corrected in `drive/mod.rs`; retain the explicit schedule-versus-seed distinction and verify both generated and handwritten reruns |
| P04 | `Outcome::ok()` checks failures while `Ending` carries separate completion evidence | Audit all callers and reports for consistent handling of budget exhaustion, stopped-early runs, and non-completion; do not infer a bug from the predicate alone |
| P05 | Trace, journal, folded dispatch order, and recent pending-work trail are separate tools | Provide a shortest-path debugging sequence and explain what each observation can establish |
| P06 | Small campaign, default tests, soak, POS comparison, and launch gate have different purposes | Provide a command/task table with expected scope; avoid presenting a small seed window as the merge gate |
| P07 | `launch-gate.sh` also runs architecture and non-fuzz suites | Explain that it is a workspace verification entry point despite its crate location |
| P08 | Known-bug benchmark copies code and reverses fixes | Keep the copy-only behavior, input tree identity, baseline, seed windows, and comparison criteria explicit |
| P09 | Seed replay depends on code, configuration, and randomness partitioning | State the reproducibility boundary; do not imply seeds survive arbitrary refactors |
| P10 | Historical measurements and current reports differ in format and scope | Keep benchmark identity, campaign boundary, configuration, and revision attached to results |

## Implementation order and acceptance

The short crate README is delivered with this audit. It is intentionally separate from this
inventory so a new reader need not first read the list of future improvements.

1. Correct current entry-point descriptions and stale navigation. Add one worked production
   regression and one rule/probe example. Keep historical coverage measurements dated.
2. Retain the implemented reproduction-policy/budget and watchdog-identity fixes; verify their
   remaining reporting paths and reconcile the arrival-order claim. Existing fixes are not
   evidence for unimplemented scheduler equivalence.
3. Improve rule/domain navigation, effective configuration reporting, and outcome categories.
4. Split large harness and recording modules only where a clear domain boundary reduces the
   amount a contributor needs to read. Preserve case semantics and compare matched seeds.
5. Simplify scheduler internals only with exact-replay, source-order, cancellation, due-wake,
   and known-bug comparison evidence.

Use these concrete checks for contributor clarity:

- A reader can explain real code versus fakes for voice and room tests from the README.
- A reader can locate, run, and interpret a production regression without reading the scheduler.
- A reader can add a rule and explain why its counterexample does not test production behavior.
- A reader can reproduce a finding with the supplied command, including policy and budget.
- A reader can distinguish a pass, an unexercised rule, a tolerated finding, exhausted budget,
  demonstrated deadlock/livelock, and a watchdog abort.
- A new input, fake, rule, or test has one obvious place to live and one obvious verification path.

Documentation checks establish navigation and command consistency. Behavioral harness changes
need targeted tests plus deterministic comparisons and relevant launch-gate coverage; readable
structure alone does not establish equivalent schedules or coverage.
