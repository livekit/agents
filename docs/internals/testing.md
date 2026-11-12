# Internal testing

Component tests, real boundary tests and deterministic voice/room simulation.

[Contributor index](README.md) · [Execution paths](architecture.md#execution-paths)

## Test boundaries

| Layer | Verifies | Dependencies and scheduling |
| --- | --- | --- |
| Core unit tests | Individual transitions, validation, effects and negative cases | Synchronous calls with supplied inputs and time |
| DST | Behavioral invariants across composed machines, failures, cancellation and event orderings | Real sans-I/O crates and the runtime's real executors, with fakes only at the SDK and provider ports: providers, the worker, the storage port (injecting every typed error), MCP transports and the room, under controlled time/scheduling |
| Async I/O and adapter integration | The shared transport, provider clients, worker IPC and supervision, storage (real SQLite and a fake AgentDB server with injected faults), telemetry delivery to the host SDK, serialization, bindings and RTC adapters | Real code against a controlled peer, real processes or the actual service |
| E2E | Complete user journeys through the deployed system | Real processes, bindings and transport; each test identifies which providers/services are live or scripted |

A contract can be exercised at several layers. DST explores its possible
orderings; adapter tests validate the real boundaries and fake assumptions;
E2E verifies that the assembled system delivers the result to the user.

For interruption, DST checks cancellation and completion orderings. E2E checks
that audio delivered through a real room stops at the caller within a specified
bound. An interrupted history flag alone does not establish that behavior.

### E2E contracts

Each test states its prerequisites, user actions, expected outcomes, forbidden
behavior and completion bounds. Assertions are standalone.
Verify business effects through recorded service calls or authoritative state,
and delivery through caller observations. Recording an event or enabling an
option does not prove the feature worked.

For approval, verify that the tool has not executed before confirmation, then
that the exact approved operation executes once and its result reaches the
caller. Recording tests verify a readable recording with expected media;
interruption tests verify playback at the receiving client.

Startup failure, missing artifacts, incomplete scenarios, empty selections,
timeouts and unexpected errors fail validation. Retrying retains each attempt
and reports instability; a later pass does not silently erase an earlier
failure. Performance measurements and exploratory recordings are reported
separately from correctness results.

### Test organization

Use ordinary Rust tests and pytest test functions, grouped by feature. A test
file contains related scenarios and assertions; reusable fixtures own worker,
room, caller and service lifecycles. Shared drivers perform actions and capture
observations without deciding whether a test passed. Fixture cleanup runs on
failure and cancellation, releases remote resources and reaps child processes.

Keep environment preparation, scenario execution, artifact capture and result
comparison separate. Command-line runners only select tests and invoke the
standard runner. Do not combine environment installation, an agent application,
caller simulation, normalization and custom pass/fail reporting in one script.
Do not build a second test framework inside a helper.

Split large files at responsibility boundaries. Keep a scenario's assertions
next to its actions; share fixtures rather than copying orchestration. Avoid
both monolithic runners and many single-use forwarding modules. Each module
must have a responsibility that a reader can name.

`tests/test_native_smoke.py` checks the assembled Python/native API with scripted
providers, local protocol peers and real job processes. Run it with
`make test TEST_ARGS="tests/test_native_smoke.py --unit"`; it also runs in the
default hermetic suite. Scripted providers live in `tests/native_smoke_fixtures.py`,
and the logging subprocess is `tests/native_job_logs.py`.

Python tests reach an offline room through the documented session, as
[programmatic tests](../testing.md#programmatic-tests) do: `session.start(agent)`
without a job, or `room=None` where a job would supply one. `_core` ships no
test-only room export. Room-level behavior is tested in Rust over the SDK-port
fakes in `lk-agents-fuzz` (`fakes::room`).
## Deterministic simulation testing

The DST is scoped to voice sessions and room I/O over fake provider and RTC ports.
Standalone A2A and TextSession execution belongs to component and conformance tests.

The harness owns every core input, executor task and fake whose ordering can
affect behavior. The DST runs the runtime's real executors (voice session, room
I/O, avatar transport, their drains and legs) over faked SDK and provider ports,
never copies of them: fakes sit only at those ports, and boundary adapters such
as the SDK room and the console's TCP port are replaced by port fakes. Executors
spawn and sleep only through the injected `Spawner` and `Clock`
([effect execution](architecture.md#effect-execution)), which the DST binds to its own
deterministic scheduler and virtual clock; every `select!` is `biased;`, so the
scheduler alone decides which ready work runs. The `rust/check` crate enforces
these rules on the runtime's `voice/`, `room_io/`, `avatar/` and `rtc/` modules, with no
baseline.

The scheduler chooses among enabled core event deliveries, executor tasks and
fake completions. Each core transition remains synchronous and atomic. Advance
virtual time to the next arrival or deadline only when nothing is enabled.

Bound total steps, steps without virtual-time progress and real execution time.
A real-time watchdog operates independently of the simulation so a
non-yielding step cannot disable it. Retain the seeds, delivered steps and last
known pending work. Report a violated contract, demonstrated deadlock and budget exhaustion
separately; a finite step limit alone does not prove livelock. Liveness checks
state the assumed progress of external services and scheduling.

### Simulated host execution

The DST harness supplies a deterministic fake of the ABI host-call contract
used by `HostInvoker`. It handles host-call effects from the real cores and
schedules correlated results, progress, stream items, failures, cancellation
and cleanup outcomes through the simulation scheduler. It stands at the
host-call port beneath the runtime's real executors and never executes
Python/Node callbacks.

Simulate stalled calls, cancellation refusal, late success, failed launch and
cleanup deadlines as well as successful execution. Preserve operation/resource
identity, source causality and declared ownership. The fake does not reproduce
agent policy: framework argument preparation, approval, recovery and continuation
still run in the real shared cores. Model application-specific validation as a
host outcome when it requires user code.

Use the fake to explore decision ordering and resource disposition. Verify real
ContextVar/async-context behavior, signature binding, application validation and
host object/task release through the actual `HostInvoker` adapters. Keep those
conformance cases aligned with the fake's contract so simulation assumptions are
checked at the real boundary.

### Scheduling boundaries

Cores receive events and time and retain no spawner or executor. The
simulation covers voice decisions, the shared agent execution used by voice,
media processing and the runtime's voice/room executors with realistic fakes.
Provider, host, voice storage and SDK room ports are fakes. Standalone A2A,
TextSession, worker, storage backends, evaluation judges, delegation transports
and MCP transports are tested in their owning components. Boundary adapters
at fake ports are outside the simulation.
They are verified by conformance tests through the real adapters with
cancellation injected at every await point, stress tests (stalled consumers,
full queues, close under load), real-peer adapter tests and a regression test
per bug.

Campaigns report which cores and fakes they simulate. Control randomness inside
fakes as well as the scheduler: supply, seed or record those choices and vary
them across campaigns. Identical inputs, build, configuration and seeds
reproduce the schedule.

### Partial-order sampling

The scheduler uses basic partial-order sampling from
[Yuan, Yang and Gu, CAV 2018](https://doi.org/10.1007/978-3-319-96142-2_20):

1. Assign a random priority to each pending step: a core event delivery, an
   executor task woken through the injected `Spawner` or `Clock`, or a fake
   completion.
2. Dispatch the highest-priority enabled step, preserving source causality
   and due-wake precedence. A source's next step receives its own random
   priority.

The harness keeps no per-step read/write footprints and does not reassign
priorities after races.

The harness varies external event order, including decision/data-plane hops,
same-instant races and bursts before settlement. Internal machine signals keep
their defined order. Effect completions return as scheduled events; they do
not mutate cores from callbacks.

An arrival-order baseline runs alongside. Campaigns record the policy with each
seed and compare policies on the known-bug benchmark. Claims about explored
interleavings apply to these scheduled steps, not arbitrary instruction
interleavings in the whole program.

Test the scheduler against small programs with independently known legal
orders. Cover source causality, due-wake precedence, wake/cancel/completion and
exact replay. Benchmarks and known-bug hit rates supplement these checks; they
do not establish scheduler correctness.

### Fakes and findings

Fakes reproduce real boundary behavior: calibrated LLM chunk timings, TTS audio
proportional to text, realtime probe fixtures and SDK room lifecycle behavior.
The measured provider timings are in `rust/crates/lk-agents-fuzz/calibration.json`, for
reference; each fake hardcodes the deciles it draws from. Fix the
fake's missing behavior whenever a real-room failure reveals a gap.

A seed reproduces a run for the same build, configuration and scheduling policy.
Record those inputs and the delivered event trace. Fixed findings become
handwritten regressions. Key distinct findings by rule and subject kind, never
generated IDs. Compare both findings and reach on the same seeds when changing
scheduling or fakes.

The [launch gate](../../rust/crates/lk-agents-fuzz/launch-gate.sh) must run before
every merge. It covers affected subject cores and runtime/binding contracts,
reports omitted coverage and rejects empty runs. Workspace compilation is not
behavioral coverage of any component; the worker and the other async I/O crates
run their real-transport, real-process and controlled-peer suites in the gate.
Its `KNOWN_FINDINGS` baseline may only shrink. The purity Clippy policy rejects core
spawns, `select!`, clocks and unbounded channels; `rust/check` rejects executor
dependencies in the cores and Tokio runtime uses and unbiased selects in the runtime's
executor modules, with no baseline.

### Observations and replay

Rules read journal observations and public outcomes, not private state. The
voice trace vocabulary is [trace.rs](../../rust/crates/lk-agents-voice/src/trace.rs): the
voice core emits each step as an effect in its output, and the runtime hands it to the
observer a test or the DST installed (production installs none).
Cover raw argument validation, pending answers, time ordering, bursts, resource
lifecycle and host-view consistency. Normalize only differences irrelevant to
the asserted contract; preserve the identities, ordering, timing and errors
needed to detect duplicate work, stale outcomes and deadline violations.

Export observations without retaining an unbounded in-memory call log.
Production replay feeds recorded events and times to cores without repeating
external effects. Recording and replay require sufficient input data, not just
an output journal. Apply retention, redaction and access policy to logs that
contain conversation text.

## Running the suites

From the repository root:

```bash
# Shared invocation/routing decisions and voice coordination.
cargo test --manifest-path rust/Cargo.toml --locked -p lk-agents-core -p lk-agents-voice

# Real runtime drivers and local-peer integration; the MCP peer (tests_new/host/mcp_peer.py)
# needs the repo Python.
MCP_TEST_PYTHON="$PWD/.venv/bin/python" cargo test --manifest-path rust/Cargo.toml \
  --locked -p lk-agents-runtime --features media-ffmpeg,media-images

# Python contracts linked against the freshly compiled native module.
uv run --no-sync cargo test --manifest-path rust/Cargo.toml -p lk-agents-py --features ffmpeg

# Same execution vectors through direct, TextSession and VoiceSession, in a fresh process.
PYTEST_ADDOPTS="$PWD/tests/test_shared_execution.py --unit -k test_shared_execution" \
  uv run --no-sync cargo test --manifest-path rust/Cargo.toml \
  -p lk-agents-py --features ffmpeg --test py_bindings

# Voice/room composition regressions with fake SDK/provider ports.
cargo test --manifest-path rust/Cargo.toml --locked -p lk-agents-fuzz --test regressions

# Full compile/lint/boundary/codegen gate; this is not a substitute for running tests.
make rust-check
```

Plain `cd rust && cargo test` does not run runtime's own tests or the Python binding contracts:
both crates are omitted from workspace default members. Runtime being compiled through the
DST dependency does not run its separate test targets. See [local builds](development.md#local-builds)
for interpreter selection, features and private target directories in parallel worktrees.
