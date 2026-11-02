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
    -> report outcome and rerun     survey_tests.rs and drive/finish.rs
```

The generator chooses situations. Fakes implement controlled boundary behavior. Production
code makes framework decisions. Rules judge observations after the run. Changing the
generator must not change how the same case is judged.

## Voice and room simulations

| Tests | Real code exercised | Controlled boundary |
| --- | --- | --- |
| Voice | `VoiceSession`, its runtime loop, room/avatar executors, and composed decision cores | Provider, host, room, and media ports |
| Room | Room I/O, participant streams, output, readiness, and lifecycle | The same fake RTC ports, including failures and stalls |

Standalone A2A and TextSession execution is outside this crate. Their component tests remain
in the core, runtime and storage crates. The voice campaign does not exercise real provider
sockets or Python callbacks. [boundary.rs](src/boundary.rs) describes the simulation boundary
and the separate conformance suites. Shared AgentRun behavior remains covered when used by voice.

## Where to start

| Task | Read or edit |
| --- | --- |
| Understand a handwritten production regression | [speech_playback_tests.rs](src/speech_playback_tests.rs), included by [regressions_tests.rs](src/regressions_tests.rs) |
| Add an action or provider delivery | [input.rs](src/input.rs), then [drive/apply/](src/drive/apply/mod.rs) |
| Change session construction | [drive/start.rs](src/drive/start.rs) and `Setup` in [drive/mod.rs](src/drive/mod.rs) |
| Change which situations are explored | [search/](src/search/mod.rs) |
| Add a behavioral contract | [rules/](src/rules/mod.rs), [check.rs](src/check.rs), and [authored.rs](src/authored.rs) |
| Understand observations and resource identity | [record/](src/record/mod.rs) and [record/views.rs](src/record/views.rs) |
| Understand scheduling | [sched.rs](src/sched.rs), then [drive/queue.rs](src/drive/queue.rs) and [drive/executor.rs](src/drive/executor.rs) |
| Understand cleanup and outcomes | [drive/finish.rs](src/drive/finish.rs) |

A rule's accepted examples and rejected counterexamples test the checker. A probe exercises
production behavior and proves that the rule's prerequisite is reachable. A rule making zero
expectations is unexercised, not a demonstrated pass. `build.rs` discovers voice rule files;
the registry checks their examples, counterexamples, and production probes.

For a worked regression, rule/probe example, fake port contracts, domain rule index and
debugging sequence, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Run and reproduce

From the repository root, after the normal development setup:

```bash
# Default tests, including checker examples and production regressions. Campaigns are ignored.
uv run --no-sync cargo test --manifest-path rust/Cargo.toml --release -p lk-agents-fuzz

# One voice seed. Preserve the original budget and any explicit policy override.
VOICE_SEED=4372 VOICE_BUDGET=3000 VOICE_POLICY=TaskBridge \
  uv run --no-sync cargo test --manifest-path rust/Cargo.toml --release \
  -p lk-agents-fuzz --test survey -- --ignored --nocapture

# A small voice campaign. Omit VOICE_POLICY to use each seed's selected policy.
VOICE_SEEDS=8 VOICE_BUDGET=3000 \
  uv run --no-sync cargo test --manifest-path rust/Cargo.toml --release \
  -p lk-agents-fuzz --test survey -- --ignored --nocapture
```

`Arrival` uses the baseline source ordering, `BasicPos` samples event-source and task-pool
turns, and `TaskBridge` also samples individual bridged task turns. These are scheduling
policies, not three different generators. Neither sampling policy explores every possible
interleaving. Core transitions and their internal signal order stay synchronous.

Add `VOICE_TRACE=1` for applied inputs and `VOICE_JOURNAL=1` for journal output. Reproduction
requires the same code and configuration; a seed is not a permanent replay artifact.
`VOICE_HANG_S` controls an independent real-time watchdog. It aborts a stuck process; real
time does not decide behavioral correctness.

## Read a result

Keep contract violations, demonstrated deadlock/livelock, and budget exhaustion distinct.
`Outcome` retains failures, expectation counts, reached journal variants, step pairs, and
input/phase observations. Reaching a journal variant does not prove a contract was exercised.

[known_findings.tsv](src/known_findings.tsv) records tolerated open findings by contract and
subject kind. A campaign passing its gate can still contain known violations. Keep that
baseline shrinking; turn fixes into production regressions.

The [launch gate](launch-gate.sh) runs a much broader suite than a local seed or small campaign.
It needs the repository environment and cargo-nextest. The [known-bug benchmark](bench/known_bugs.sh)
compares search policies against reverted fixes in a copied tree.

For design detail, read [TEST_STRATEGY.md](../lk-agents-voice/TEST_STRATEGY.md).
[COVERAGE.md](COVERAGE.md) contains dated measurements, not a live capability inventory.
The [fuzz crate audit](../../../docs/internals/rust-fuzz-inventory.md) inventories simplification
and onboarding improvements alongside the
[transition inventory](../../../docs/internals/rust-transition-inventory.md).
