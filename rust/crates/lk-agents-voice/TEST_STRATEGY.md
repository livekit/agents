# Voice test strategy

This document defines the end state to implement. It is not a description of completed
coverage. Product decisions still needed are collected at the end; no timeout values
or event-wait semantics are silently assumed.

## Reference material and starting point

Build the replacement test suite from the contracts and architecture in this document.
The previous voice tests are archived at
[reference/voice-tests](../../../reference/voice-tests), outside the Rust workspace.
They include the old harness, rules, inline-test source snapshots, corpus, fuzz inputs,
and campaign scripts. The archive's manifest records original paths and content hashes.
It is authoring material, not a runnable replacement suite or an acceptance baseline.

Use that archive and the Python reference at `~/livekit/agents` only to investigate
behavior and possible cases while writing new contracts. Neither defines correctness
or dictates the replacement layout. No reference structure or type is selected for reuse by this document. Any proposed
reuse needs an explicit justification against the new boundary and contract; a familiar
name or existing implementation is not that justification. Do not import archived helpers, require old tests to
pass, or depend on reference source files during new test execution.

The suite that implements this document is the workspace crate `lk-agents-fuzz`
(`rust/crates/lk-agents-fuzz`). Its module docs describe the layout; `cargo test -p
lk-agents-fuzz` runs the contract registry, probes, saved cases and a bounded fresh
campaign, and `cargo voice-report <artifacts dir>` prints what a campaign reached and found.
Compilation checks do not establish runtime correctness; the acceptance evidence in §9 does.
Other crates' tests are outside this voice suite.

## 1. Objective and agreed decisions

Find unknown interactions across the supported APIs, callbacks, providers, media,
and lifecycle. The generator must not require us to name a dangerous combination
before exploring it. Targeted scenarios supplement broad exploration.

| Decision            | Agreement                                                                                                        |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Priority            | Missing progress first, incorrect behavior second, ownership/lifecycle third; all are required                   |
| Correctness         | Explicit behavioral contracts; Python may resolve ambiguity during authoring, never act as a runtime test oracle |
| Inactivity          | Per work operation, starting when requested, including queue time; unrelated activity does not reset its timer   |
| Continuing progress | No blanket total-duration limit on work that continues making meaningful progress                                |
| Provider failures   | Test retries, recovery, tolerance, counter resets, and eventual session closure                                  |
| Internal stalls     | Detect missing progress independently of provider error reporting                                                |
| Exploration         | Combine unstructured generation, observed-state weighting, and controlled scheduling                             |
| State weighting     | Use measured coverage gaps to guide additional exploration; keep supported calls available in awkward states     |
| Readability         | One observable guarantee per rule; setup, action, and assertion understandable together                          |

Exact progress events, inactivity values, expiry handling, and pure event-wait
semantics still need decisions. Tests must not assume these already exist in the
runtime. No finite campaign establishes that every possible interaction is covered.

State weighting is an exploration optimization, not a model of likely production
behavior. Measure which overlaps and contract prerequisites broad generation reaches,
then adjust action probabilities to reach underexplored combinations. Weight inputs,
never expected outcomes. Use actual pending calls and observed events when applying
those preferences; do not infer state from the actions the generator merely attempted.

Keep a broad campaign without state-based preferences. Compare weighted and unweighted
campaigns under the same execution budget, recording prerequisite/overlap coverage and
execution counts. Retain weighting only where measurements show its benefit, and keep
the broad campaign to expose interactions our preferences overlook. No supported API
call becomes unavailable merely because its timing is awkward or it may be refused.

## 2. What catches what?

| Failure                                            | Required check                                                                                                  | Search mechanism                                                                                     |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| An API call never acknowledges                     | Retain its actual receiver/result; require its contractually due response                                       | Broad overlapping calls, including startup and closure                                               |
| Accepted work stops progressing                    | Per-operation inactivity checks, before teardown                                                                | Independent provider/callback/output delays, failures, holds, and releases                           |
| Wrong output or missing output                     | Compare with independently specified inputs and expected outcomes                                               | Broad contracts plus targeted cases with known expected results                                      |
| Wrong-owner or duplicate result                    | Match invocation, request, attempt, generation, and resource identities                                         | Multiple outstanding operations, replacements, retained handles, late delivery                       |
| Output precedes its prerequisite                   | Require the matching authorization or other contract-defined prerequisite to precede the actual output boundary | Vary prerequisite, content readiness, invalidation, and playback ordering                            |
| Cancellation races with completion                 | Check the independently specified allowed outcomes and subsequent effects for that operation                    | Release cancellation and completion in both orders and in bursts before processing                   |
| Late results affect replacement work               | Require effects to belong to the correct request, generation, or resource instance                              | Deliver old results before/during/after replacement and cancellation                                 |
| Overlapping operations block each other            | Enforce each operation's own progress obligation; unrelated activity cannot reset it                            | Keep multiple calls pending; vary dependency release order, including holding unrelated dependencies |
| Completion races with closure                      | Check caller settlement, at-most-once completion, and resource cleanup against the close contract               | Race completion with close, repeated close, and timeout delivery                                     |
| Incorrect retries/tolerance                        | Start from the fake's injected result; verify each attempt, error, reset, and closure                           | Independent per-request failure/recovery decisions                                                   |
| Cancellation or closure leaks work                 | Verify settlement and owned-resource cleanup; distinguish watchdog abort                                        | Cancellation/close during every reachable phase                                                      |
| Runtime panic or executor stall                    | Surface failed tasks; independent process watchdog                                                              | All execution entry points                                                                           |
| Checker accepts an invalid trace                   | Small local positive and negative controls                                                                      | Small authored observation records beside the invariant                                              |
| A known failure returns                            | Replay a saved case and require its original obligation to be exercised                                         | Regressions plus fresh variations                                                                    |
| An unknown interaction violates a general contract | Apply all relevant progress, ownership, and behavior checks                                                     | Broad property generation and coverage-guided native fuzzing                                         |

Races are exercised through scheduling; contracts detect their incorrect consequences.
For competing events whose protocol permits either order, explore A before B, B before
A, and both released before the runtime processes them. Include shared timestamps and
before/at/after deadline boundaries. Record actual boundary observations: selecting a
delivery order does not prove the corresponding processing order occurred. Required
ordering probes must verify their intended order was reached.

These examples do not restrict generation to named event pairs. Generate overlapping
operations and higher-order event combinations across the supported vocabulary, and
apply all applicable contracts. Specify allowed outcomes independently; some races
legitimately have more than one outcome. Never require a fixed winner merely because
one execution produced it.

Coverage requires both executable scheduling controls and a check of the affected
behavior. External delivery ordering does not establish coverage of every internal
task poll or thread interleaving. Report those scheduling limits as specified in §5;
never report external-event exploration as exhaustive concurrency coverage.

A local counterexample validates a checker, not runtime behavior. A generated input
only contributes behavioral coverage when its prerequisite and assertion actually
occur. Neither code coverage nor a whole-rule Pass is proof of completeness.

## 3. Shared execution and generation boundaries

Use the real Rust session with controllable dependencies. Share execution between
broad exploration, targeted tests, and saved regressions. Do not build one simulator
per rule or copy production decision logic into the generator.

The harness controls execution and records what happened. Rules state whether that
behavior was correct. The runner chooses or replays inputs and invokes those rules.
Generation, scheduling choices, coverage, and shrinking are functions used by the
runner, not additional architectural layers.

These techniques control different parts of exploration and can be combined in one
execution. The examples below illustrate their mechanics; they are not an exhaustive
list of scenarios or the only places the rules apply.

| Technique                                    | Exact choice it controls                                                                                                                                                                   | Concrete example                                                                                                                                                                                                                                                       | Required safeguard                                                                                                                                                                      |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Broad random action selection (unstructured) | Select the next action and arguments from the supported vocabulary without a scenario template or state-based preferences. Reference actual handles/requests when the action requires one. | Draw `generate_reply`, another `generate_reply`, non-forced interrupt, then close; issue the calls without waiting for earlier results. This sequence is a possible draw, not a prescribed pattern.                                                                    | Keep all supported host calls selectable in awkward states, including calls expected to be refused. Random selection does not mean malformed inputs or necessarily equal probabilities. |
| Observed-state weighting                     | Change action probabilities using actual observations to address measured coverage gaps. It changes selection frequency, not allowed outcomes.                                             | Measurements show little interruption during playback. Once output observations establish playback, increase interruption's selection weight; continue allowing generation, delivery, close, and other supported actions.                                              | Compare coverage against broad selection under the same budget. Do not infer playback merely because `generate_reply` was issued or remove other calls from consideration.              |
| Controlled scheduling                        | Choose when calls are issued, dependency results are released, specific acknowledgements are awaited, and virtual time advances, within the documented scheduler model.                                   | With a provider result held, explore completion before cancellation, cancellation before completion, and both delivered before the next poll. Also vary their timing around a configured deadline.                                                                     | Verify the actual observed ordering for required probes. Respect protocol causality; report internal task/thread ordering the harness does not control.                                 |
| Targeted prerequisite setup                  | Supply a deliberate input prefix that reliably reaches a particular contract's prerequisite; optionally generate further actions after it.                                                 | Establish a turn-policy condition that forbids speculative playback, let the speculative provider produce audio, and verify audio readiness. Check that playback remains blocked, then supply the permitted authorization prerequisites and check progress separately. | Fail the targeted test if the prerequisite was not reached. Run all applicable rules through the shared execution path; do not restrict broad generation to this setup.                 |
| Journal-guided search                        | Keep cases that produced a driver journal variant no earlier case produced, and mutate their action lists. The journal, not line coverage, is the feedback: every `Step` family carries the `Vocabulary` derive, so the never-produced variants are known and reported. | A mutated case makes a cancelled request's late result reach a journal variant no case had produced. The case is kept and mutated further; the ownership rule checks whether the late result incorrectly affects newer work.                                          | A new journal variant is a search signal, not a correct/incorrect verdict. Cases stay plain recorded actions, replayable through the same runner. A native fuzz target (`FuzzSampler::from_bytes`) can drive the same choices if this plateaus. |

For example, one execution can start with a targeted prefix, use state-weighted choices
for subsequent actions, and vary delivery order through controlled scheduling. Native
fuzzing can drive those choices from its input bytes. Record the choices actually made
so replay does not depend on choosing the same random actions again.

### Generate connected interactions; check their outcomes

Random selection operates on meaningful actions attached to real interactions. It does
not produce unrelated lifecycle events. For example:

```text
request speech → observe its actual handle and provider requests
               → choose content, delivery, stream end, and output acknowledgements
               → observe and check that speech's terminal outcome
```

Maintain only the actual references needed to execute these actions: issued call IDs,
returned handles, pending provider requests, callback invocations, and output operations.
A provider delivery targets a request that exists. A handle operation targets a handle
actually obtained, including retained older handles. This is dependency bookkeeping,
not a model predicting what the session should decide next.

| Choice                  | What the generator does                                                                                                          | What the rule checks                                                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Continue an interaction | Supply finite content to the actual request, end its stream, and acknowledge the corresponding output operations when they occur | The associated speech completes under its contract once its prerequisites are satisfied; acknowledgements alone do not establish completion |
| Delay a dependency      | Keep one real request or output operation pending while issuing other calls, delivering other results, or advancing time         | The correct operation's progress/deadline policy is enforced; unrelated work remains independent where required                             |
| Disrupt an interaction  | Interrupt the actual speech, fail a request, replace an output, or close while work is pending                                   | The permitted cancellation, failure, recovery, and settlement outcomes occur for the correct identities                                     |
| Revisit older work      | Operate on retained handles or attempt protocol-permitted late delivery to an earlier request                                    | Old work cannot affect a new owner; repeated operations receive their contractually permitted results                                       |

A fake output's playback-complete notification can be a generated dependency input.
The session's speech-completed notification or completed speech handle is an observed
product outcome. Never inject that outcome to make the interaction appear complete,
or infer it from having supplied all the inputs. A missing terminal outcome must remain
observable and fail its due obligation, including when no completion event was emitted.

Provide continuation choices alongside interruption/failure choices at each applicable
boundary. Required targeted probes must complete a finite interaction normally as well
as exercise its disrupted outcomes; broad search mixes continuations and disruptions
across multiple outstanding interactions. Continuing one interaction must not become
an atomic scenario that settles it before another action can be issued. New dependencies
created during continuation are observed and addressed on subsequent runner actions.

Measure whether generated continuations actually reach terminal outcomes and whether
disruptions occur with work outstanding. Do not claim useful generation from merely
selecting actions. Supported host calls remain available in awkward states; dependency
references constrain where an input can be delivered, not when a host is allowed to call.

### Preserve the search space when adding continuation logic

Coherent continuation adds choices that reach deeper behavior; it must not replace
broad action selection with a sequence of fully settled interactions. Every hard
restriction on generated actions must identify the actual interface or protocol
constraint that requires it. Put that justification beside the restriction and in the
executable boundary inventory. “Unlikely,” “inconvenient,” and “not the expected next
step” are not interface constraints. Campaign masks are explicit search configuration,
not claims that the omitted actions are invalid.

| Justified dependency constraint                                 | Unjustified restriction                                         |
| --------------------------------------------------------------- | --------------------------------------------------------------- |
| A result delivery references an actual request                  | Only the newest request may receive a result                    |
| A handle operation references an obtained handle                | Completed, cancelled, or older handles cannot be selected       |
| Preserve ordering guaranteed by the specific provider interface | Cancellation always removes every possible late-delivery action |
| Offer actions that finish an interaction                        | Finish it before issuing another operation                      |
| Match each input to its actual owner                            | Allow only one outstanding interaction                          |

Retain multiple outstanding operations, repeated calls, old handles, partial progress,
repeated recovery, held dependencies, and permitted late/duplicate delivery. Independent
events can be reordered or released in bursts, including around deadlines. Supported
host calls remain selectable during startup, replacement, cancellation, and shutdown.
For closed receivers, retain permitted delivery attempts and record rejection; do not
bypass the real interface to force acceptance.

Validate these properties in generator capability tests using controlled observations:
adding continuation choices must leave the broad eligible action set intact. Inspect
eligibility directly rather than relying on a random sample to demonstrate absence of
filtering. Deterministic probes must also execute older-handle operations, overlapping
requests, and applicable late-result attempts through the harness; enumeration alone
cannot prove that execution works. Compare measured reachability before and after
weighting or continuation changes under the same bounds.

Classify genuinely malformed protocol inputs separately and exercise them through the
real boundary capable of receiving them. Record their rejection/robustness contracts
and execution-layer owner. Neither silently exclude them from the inventory nor inject
impossible private-state transitions into the session. Coverage claims must distinguish
valid-interaction exploration, malformed-input robustness, and unsupported boundaries.

A masked campaign intentionally omits some operations to reach deeper states before
close/interrupt cuts them short. Keep it alongside unmasked exploration, not instead
of it. Vary sequence length, outstanding-operation count, and timing independently.
Do not limit exploration to pairs; higher-order and long-history failures matter.
These are composable choices, not five separate runners: unstructured/state-aware
selection chooses inputs, scheduling chooses delivery, and property generation or
native coverage guidance drives the search. A targeted scenario fixes prerequisites
while using the same controls and checks.

### Code layout

```text
runner.rs       choose or replay actions → execute → check → save failures
harness.rs      real session, controlled dependencies, pending results, observations
rules/          one readable guarantee per file, with its local tests
replay/         recorded inputs for the same runner
```

Start with these pieces. Case data and action generation live with the runner; they
do not need their own modules, traits, or public interfaces. The harness contains its
fakes until a concrete dependency is substantial enough to justify a separate file.
These are proposed new boundaries, not instructions to copy reference modules.

The runner calls the harness to act and borrows its observations for the rules.
The harness does not decide expected outcomes; the rules do not drive the session.
A targeted test supplies inputs through the same runner and verifies that its required
prerequisites were actually reached. Generation never consumes rule verdicts to decide
which behavior should be allowed. Journal-guided search and any native fuzz entry point are
thin callers of this same execution path.

- Preserve one independently named contract per rule file, as agreed, with its local
  positive/negative tests at the end. Do not split its prerequisite, assertion and
  counterexamples across helper modules.
- Category files are short indexes or campaign registration, not a second home for
  the contract's business logic. Simple table-driven cases stay beside their rule.
- Inline a helper used by only one contract. Extract shared code only when multiple
  concrete callers need the same behavior, with the same semantics.
- Keep shared helpers about mechanics: create a provider, deliver an event, find a
  result by ID. Helpers must not hide why a contract passes or its exceptions.
- Avoid generic Rule/Scenario/Policy/Engine layers that duplicate responsibilities,
  speculative plugin interfaces, and builder chains that obscure the execution.
- Prefer concrete structs/enums and ordinary functions. Add a trait only for an actual
  boundary with distinct implementations, not merely to give a responsibility a name.
- Avoid both tiny forwarding files and a giant harness owning unrelated concerns.
  Split a large file only along independently understandable responsibilities, such as
  provider fakes versus execution, not arbitrary line counts.
- Keep recorded inputs under `replay/`; it is data executed by the shared harness, not another
  test implementation. Keep category navigation in code rather than another document.

Acceptance: a reader can open a contract file and explain its prerequisite, assertion, and any targeted seed; inspect the harness to understand execution; and inspect the campaign
runner to understand search/replay. They should not need to chase a rule-specific chain
through all three to determine what the test promises. No hard file-count target is
imposed, but every new shared module must identify its concrete callers and ownership.

### Independent evidence

| Evidence source                        | What it establishes                                                    | What it cannot establish alone                           |
| -------------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------- |
| Test-issued API calls                  | Intended input, issue time, caller identity and required result        | Acceptance or completion                                 |
| Controlled provider/callback           | Actual request received, result made available, delivery/release order | Whether the runtime consumed or acted on that result     |
| Actual result receiver/output boundary | Caller-visible acknowledgement, output, error or completion            | Whether that outcome was permitted by the contract       |
| Runtime trace/state                    | Diagnostic context, internal transitions and correlation information   | Its own correctness, or delivery to an external observer |

Every contract names its independent prerequisite and outcome observation. A provider
result being ready and the runtime consuming it are distinct observations. Correlate
both when required. Instrumentation must not silently serialize the race being tested.

Use test-owned invocation/dependency identities and associate runtime-generated IDs
when actually observed. A missing association is unresolved evidence, not permission
to skip a required outcome. If a guarantee requires an unavailable observation, mark
that capability missing and add it before claiming coverage. Never synthesize expected
behavior from the production decision function or trust a journalled answer as proof
that the caller received it.

### Search recipes and recorded executions

A search recipe describes available operations, parameter domains, biases, bounds,
and any fixed seed prerequisites. During execution, adaptive selection can refer to
actual pending requests, retained handles, and observed state.

A recorded execution contains the concrete initial configuration, each issued call,
actual identities and associations, each controlled dependency result, scheduling
choices, virtual times, and the failed obligation. Record decisions as they happen,
before shrinking. A random seed or recipe alone is not a replay artifact.

Replay follows recorded choices using causal references such as “second request from
provider X,” with validated identity associations, rather than assuming runtime IDs
will be numerically identical. If a referenced request does not occur, report replay
divergence. Do not silently select another request or regenerate a different choice.
Shrinking removes or simplifies concrete events while repairing only valid causal
references and preserving the original failed obligation. Keep recipes and original
recordings even when a smaller reproducer is found.

### Required harness behavior

The shared execution path must retain pending callers, support issuance without an
implicit settle, and independently release provider/callback/output dependencies.
It must support incremental streams, old and pending handles, live configuration
changes, startup and closure, and callbacks issuing supported APIs. Pipeline and
realtime adapters share accounting while retaining their protocol-specific behavior.

Record actual overlap, meaningful progress, result routing, and resource lifetime.
Check normal progress before teardown and distinguish normal completion from cleanup
forced by watchdog/abort. Process watchdogs must work even if an executor poll blocks.
These requirements apply across the inventoried vocabulary, not just a named example.

## 4. Minimal runner API

This is a design sketch, not implemented code. The harness records what crosses
actual boundaries and keeps real pending handles. It does not build a second model
of replies, turns, provider state, or the session lifecycle.

### What is available when writing a rule?

A rule takes only the records it needs. There is no normalized `Run`, reconstructed
`Reply`, or universal `Evidence` object.

| Data available                             | Implementation boundary                                                                                                           |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Issued actions, arguments, and issue times | Append when the runner performs the action                                                                                        |
| Actual API results                         | Keep the real receiver/handle; record its result when polled                                                                      |
| Provider requests and supplied responses   | Append in the fake's request/delivery methods, using the actual request payload                                                   |
| Output calls and their results             | Append in the output fake at the defined boundary                                                                                 |
| Host notifications                         | Record what the actual consumer receives                                                                                          |
| Internal decisions required by a contract  | Record the specific decision at its occurrence, with its subject identity; select the observation interface during implementation |
| Virtual time                               | Read the controlled clock                                                                                                         |
| Resource creation and release              | Add direct boundary counters/records only where a concrete contract needs them                                                    |

Actual public API payloads are the inputs to the system under test, not a mandate to
copy its internal types into the test architecture. Choose observation records from
the facts the contracts require.
Add test-owned IDs for actions or fake requests only where a replayable identity is
needed. Correlation means recording an actual association, not guessing from “the
current reply.” Do not copy every production type merely to make it serializable;
encode only the input and diagnostic data needed for reproducible cases.

### Rule signatures expose their inputs

```rust
fn preemptive_audio_requires_authorization(
    authorizations: &[AuthorizationObserved],
    output: &[AudioOutputCall],
    check: &mut Check,
);

fn commit_returns_the_expected_transcript(
    issued: &[IssuedCommit],
    returned: &[CommitResult],
    supplied_transcripts: &[SuppliedTranscript],
    check: &mut Check,
);
```

These names describe proposed boundary records, not selected reference types. For this
example, `AuthorizationObserved` needs a generation identity and occurrence stamp;
`AudioOutputCall` needs the matching identity, occurrence stamp, and actual output
operation. The producer also records whether a generation was speculative when it is
created. The implementation must establish how these identities cross the boundaries.
No archived trace enum or event hierarchy is required. Start with the records needed
by the first contract rather than declaring an exhaustive internal-event model.

For the preemptive rule, the necessary facts are narrow: the output's actual generation
identity, whether that generation was speculative, its matching authorization event,
and their causal order. The rule reads the recorded authorization decisions and actual output calls. It does not construct or maintain a reply object. If the driver/output
boundary lacks that identity or origin, add narrowly scoped instrumentation or report
a missing observation; never reconstruct it from a guessed lifecycle.

`Check::expect(condition, subject, message)` records an assertion and its result.
It does not decide prerequisites or policy. No assertions means the rule was not
exercised. Missing instrumentation is unsupported, not an empty passing record set.
A tiny category wrapper supplies the required slices to its rule functions; do not
add a dependency-injection framework or force every rule through a universal input type.

Keep a single-use observation scan inside the rule. Extract a helper only after concrete
rules need the same mechanical lookup, such as finding an event for an exact ID.
A helper must not compute whether authorization was legal, whether a turn ended, or
whether cancellation should win. Those decisions belong in explicit contracts.

Output timing needs real ordering evidence. A timestamp or the order in which two
logs were drained is insufficient. Prefer synchronous stamps at the actual boundaries
using one shared monotonic occurrence counter in controlled single-thread execution.
For concurrent execution, use synchronization-backed causal observations; an atomic
counter alone does not prove a happens-before relationship. Do not invent a complete
causal-history model to support one assertion. If the selected execution mode cannot
establish the needed order, report that limitation explicitly.

### Cases are input, not another runtime model

```rust
struct Case {
    setup: Setup,
    actions: Vec<Action>,
}

enum Action {
    Call { id: CallId, operation: Operation },
    Deliver { request: RequestId, response: Response },
    Answer { callback: CallbackId, answer: CallbackAnswer },
    AdvanceTime { to: Ts },
    Await { observation: ObservationId, deadline: Ts },
}
```

These enums describe concrete external actions. Their order specifies external
scheduling: consecutive calls/deliveries form a burst; `Await` waits for a named actual
result or notification under a contract-owned virtual deadline. It does not establish
global quiescence. Ordinary Tokio polls tasks; no custom poll budget or yield loop is
part of the case API. The process watchdog handles blocked polls.
Add further scheduling actions only when their execution mechanism is implemented and
its scope is documented.
They do not represent derived reply states. `Operation` grows from the actual API
inventory, not hypothetical future adapters. Generation settings describe domains,
weights, and bounds separately; a saved case records the choices actually made.

Save the case with the raw observations, configuration/code version, scheduler model,
and original failure identity. Targeted seeds carry explicit required observations
checked by their test. They do not define which global rules run. Replay executes
recorded actions and validates their references instead of regenerating random choices.

### One harness and one execution loop

The harness owns the session, controllable fakes, real pending calls, clock, and direct
recordings. This is the necessary operational bookkeeping; no parallel domain model.

There is no `Generator` interface or separate case-data layer. `choose_action` is an
ordinary runner function that reads actual pending requests and observations. Replay
reads the next saved action instead. Both use `execute_action` below.

```rust
// runner.rs — uses Case and Action defined above.
// Result here carries execution/check failures.
fn choose_action(rng: &mut Rng, settings: &SearchSettings, h: &Harness) -> Action;

fn execute_action(
    h: &mut Harness,
    action: &Action,
    recording: &mut Recording,
) -> Result<()> {
    h.validate(action)?;            // Does the referenced request/handle exist?
    recording.append_and_flush(action)?; // Preserve input before a blocked poll.
    h.apply(action)?;               // Calls retain receivers; only Poll runs tasks.
    check_all(h)?;                  // Includes due per-operation progress checks.
    Ok(())
}

fn check_all(h: &Harness) -> Result<()> {
    // Ordinary calls passing the observations each rule needs.
    // Collect assertion counts and failures without putting policy in dispatch.
    // The complete registry invokes every supported rule, not just selected seeds.
    /* ... */
}

// harness.rs
impl Harness {
    fn new(setup: &Setup) -> Result<Self>; // Startup remains controllable.
    fn validate(&self, action: &Action) -> Result<()>;
    fn apply(&mut self, action: &Action) -> Result<()>;
}
```

The declarations are a skeleton, not compilable implementation or selected reference
types. Define the case and actions once in the runner. `Recording` is an append-only
artifact writer, not another execution model.
`Harness::apply` dispatches concrete actions directly to the session or fake. Issuance
may synchronously produce observations; it must not implicitly poll the executor or
await acknowledgement. Advancing the controlled clock exposes due timers for a later
poll rather than silently settling work. Execution modes must document any timer order
the executor controls instead of the harness.

```text
parent starts child watchdog and retains its input artifacts
child creates a fresh harness and checks startup observations
repeat within search bounds:
    choose an action, or read the next recorded action
    execute_action(harness, action, recording)
    stop on failure, divergence, or exhausted bounds
run explicit finish/cancel actions through the same execution path
preserve the first failure; record cleanup outcomes separately
report checked/unreached/unsupported contracts and exploration coverage
```

For replay, iterate `case.actions`; for fresh search, call `choose_action` each time.
Systematic exploration supplies its chosen prefix through that same path. Shrinking
edits recorded actions and replays candidates on fresh harnesses. These are runner
functions, not interchangeable engine implementations. Keep campaign settings separate
from concrete saved actions so replay does not rerun random selection.

Record startup/step failures too. The supervisor must preserve input even if the
executor blocks. Replay, systematic exploration, and shrinking reuse this execution
path. Shrinking accepts only a case that still violates the original obligation.

### Complexity limit

Add harness code only to drive an actual boundary, record a fact, control scheduling,
or reproduce a failure. Progress bookkeeping may store an operation ID, issue time,
last contract-defined progress observation, and real pending result. It must not grow
into a second scheduler or ownership model of the driver.

Do not maintain the same fact in several observation logs and a reconstructed reply map.
Prefer a straightforward scan of a small recording over a derived cache; optimize only
when measurements justify it. If a rule requires substantial lifecycle reconstruction,
first reconsider its boundary or add the missing direct observation. If a helper is
as complex as the behavior under test, it is not an independent oracle by default.

The first checkpoint is two overlapping calls, one controlled dependency, recording
and replay, and one readable rule. Review the amount of plumbing needed to reach that
checkpoint before expanding the vocabulary. No file-count or line-count target can
replace this review, and infrastructure must not grow speculatively ahead of contracts.

## 5. Search-space inventory

Every supported boundary needs a mapping to generation, actual execution, observed
prerequisites, independent checks, and explicit exclusions. Enum coverage alone is
insufficient. New public APIs/events must update this mapping. Keep the executable
inventory with the tests, not in another design document. Each entry identifies its
boundary, arguments, permissible results, progress observations, generator, checker,
and supported execution modes. An exclusion names the missing capability and owner;
it is reported as a gap, not equivalent to coverage.

| Dimension       | Required variations                                                                                             |
| --------------- | --------------------------------------------------------------------------------------------------------------- |
| Public API      | Arguments, absent/null/empty values where supported, repeated calls, overlapping calls, dropped callers         |
| Handle identity | Pending, active, completed, cancelled, and older retained handles; multiple waiters                             |
| Lifecycle       | Startup, failed startup, agent/model replacement, drainage, close in flight, repeated closure                   |
| Providers       | Per-attempt success/failure, partial output then silence, recovery then failure, late results                   |
| Callbacks/tools | Reentrant supported operations, multiple distinct tool outcomes, callback release/cancellation order            |
| Media/output    | Chunk boundaries, incremental production, full queues, slow/disappearing consumers, partial text/audio delivery |
| Timing          | Before/at/after configured deadlines; simultaneous events; bursts without settling; variable release order      |
| Configuration   | Defaults/overrides, live updates, provider capabilities, pipeline/realtime transitions where supported          |
| State history   | Repeated swaps, holds/releases, cancellation, recovery resets, resource reuse                                   |

Do not classify late or duplicate events as invalid without consulting their protocol.
The treatment of genuinely invalid provider input remains an open campaign decision.
Random external scheduling does not cover every Tokio poll or OS-thread interleaving.

### Required capabilities and their acceptance probes

These probes turn known blind spots into acceptance requirements for the replacement
suite. They supplement open-ended generation; they do not define its available
combinations. Historical issue counts are prioritization leads, not verified coverage
statistics, and old rule IDs are lookup aids in the archive, not specifications.

| Area                         | Required generator capability                                                                                              | Independent check / minimum probe                                                                                                                                                |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Output and shutdown          | Delayed, missing, early, duplicate and failed playout notifications; blocked capture/flush/close; output replacement       | Observe the actual sink calls, each output's close completion, and pending caller results; distinguish normal cleanup from forced abort                                          |
| Realtime tools               | Advertised callable tools with real controlled bodies; independent tool/result timing and server auto-follow-up capability | Observe actual body invocation, arguments, output delivery and permitted follow-up requests; an unknown-tool error is not tool-body coverage                                     |
| Parallel tools               | Different outcomes for calls in the same step, including return, failure, cancellation and distinct handoff targets        | Match outputs and completion to call IDs; specify competing handoff behavior independently before asserting a winner                                                             |
| STT recovery                 | Drop/reopen, repeated failures and recovery later in a long session; delayed old-stream results                            | Observe actual stream creation/closure and transcripts by stream identity; check configured retry/tolerance and reset behavior                                                   |
| Turn timing                  | Independently controlled STT and VAD onset/end/final timing                                                                | Assert endpointing against the selected turn-mode contract; do not derive both streams' timings from one fixture                                                                 |
| Content inactivity           | No first content token, keep-alive-only traffic, one token then silence, continuing meaningful content                     | Separate transport activity from semantic progress; test before/at/after the configured inactivity boundary and require the specified handling                                   |
| Text segmentation            | URLs, numbers, abbreviations, punctuation, Unicode and varied chunk boundaries                                             | Use independently authored expected text/segments where contractually defined; check content conservation and streaming behavior without using the production splitter as oracle |
| Manual turn mode             | Calls and media while manual mode is active, with no commit, one commit, repeated/superseding commit and closure           | Independently specify which inputs may trigger a reply; distinguish explicit host reply calls from automatic end-of-turn behavior                                                |
| Echo and audio routing       | Explicit output-to-microphone feedback with controlled delay/gain, alongside real user input                               | Verify declared echo/interruption behavior; a content-blind VAD fixture is not echo coverage. Use the appropriate media/integration layer when simulation is insufficient        |
| Context during running tools | Next request arrives while an earlier tool remains pending; overlapping turns and late outputs                             | Inspect actual model request messages and tool-call IDs; a context hash alone cannot prove history consistency or placeholder behavior                                           |
| Frame and protocol edges     | Empty/zero-sample frames, supported rate changes, interim/final reordering and late results                                | Classify each input using the actual protocol contract; test permitted behavior and separately identified rejection cases                                                        |

Per-request controls must support, at minimum, success-then-failure, failure only on a
selected request/attempt, partial success then failure, and late delivery after
cancellation where the protocol allows it. A fail-first-N preset is useful as a seed
but is not a substitute for independent decisions on actual requests.

Also require direct capability probes for bursts of unacknowledged calls, non-forced
interrupt refusal, real commit/idle receivers, older-handle operations, hold/release,
playout waiters, and completion callbacks. Each probe records that the boundary was
actually exercised; adding an enum arm or a no-op fake is insufficient.

A process watchdog must preserve the in-flight case independently of the blocked
child. Process termination alone detects a stall but is not a reproducible finding
unless the chosen inputs and last available execution position were retained.

### Implementation contracts for the missing capabilities

The table above is an acceptance index. The following specifies how to supply those
capabilities. These are required behaviors of the new machinery, not instructions to
port an archived module. Keep them in the shared harness and its dependency adapters;
a capability does not require its own framework or file.

**Pending operations and dependency delivery.** Issuing a host call stores its real
result receiver or handle under the input call ID and returns control immediately.
An explicit runner poll collects ready results without awaiting any pending result.
Calling, polling, releasing a dependency, and advancing time are separate actions.
The generator can issue a whole burst before polling. Retain commit, idle, playout,
and completion waits, including multiple waiters on old handles. Record caller drops
explicitly; they remove that caller's response obligation, not the underlying work's
obligations. Exercise forced and non-forced interrupts independently, and floor or
interruption holds with separate release actions.

Each fake registers an invocation when the product actually calls it. Its registration
contains a unique request ID, attempt/stream identity where applicable, actual input,
and the real channel or future completion mechanism. The generator chooses each next
chunk, failure, completion, or release for that specific invocation. Never select one
behavior shared by a whole provider leg or all tools in a batch. A delivery references
an observed invocation; guessing a future runtime request ID is replay divergence.
The fake records attempted delivery and whether the receiver accepted it, separately
from any observation that the product processed it. A closed receiver is an observable
outcome, not permission to inject directly into private product state.

Acceptance: two outstanding requests receive different outcomes in reverse order;
one remains blocked while unrelated work completes; request two alone fails; a later
request recovers and then fails again; a cancelled request receives a late delivery
attempt. Each result must be attributable without using “latest request.”

**Output capture, playback, and shutdown.** Model the actual output interface's
independent operations separately: accepting a frame, reporting playback progress or
completion, flushing/clearing, and closing. A frame accepted into a queue does not
prove it played. Each operation may complete, fail, or remain pending independently.
Attach output-instance and playback identities when the real operations are issued;
replacement creates a new instance. Delay or attempt duplicate/early notifications
through the actual interface, with protocol-invalid cases identified separately.
Apply the same controls to text output and each supported adapter close handshake.

Rules inspect accepted content, actual playback reports, clear/close calls, their
results, and caller settlement. Check that old-instance notifications cannot complete
new playback, cleared content cannot subsequently be delivered as new output, and
normal shutdown satisfies its specified per-output handshake. Forced cancellation
must be reported separately from graceful completion. The prerequisite tests must
hold capture, playback notification, and close independently and then release each;
a “slow sink” flag that delays all three together does not meet this requirement.
Concrete close deadlines and failure handling come from the approved output contract,
not a harness assumption that every close must succeed.

**Realtime and parallel tool calls.** Advertise callable tools and execute controlled
bodies through the real tool interface. Register each body invocation with its call ID
and actual arguments. Each body independently returns, errors, requests handoff, or
remains pending; cancellation and late completion are separate choices. Generate mixed
outcomes within the same response, not just across successive responses. Record the
actual tool results sent to the provider and actual follow-up requests with their cause.

Check invocation and result identity, absence of duplicate results, and follow-up
ownership separately. Server-managed continuation and client-requested continuation
need distinct contracts; unrelated host reply requests must not count as tool
continuations. Required seeds include return plus handoff, two distinct handoffs,
failure plus success, cancellation plus late result, and both realtime continuation
capabilities. The winning-handoff and continuation-join policies require explicit
allowed outcomes before their rules are implemented; never pick the first observed
winner as the expected result.

**STT recovery and independent turn signals.** Give each opened STT stream its own
input recording and controllable result producer. Disconnect ends that stream; reopening
must be an observed product action, not a fake silently replacing its channel. Keep old
stream producers addressable for permitted late-delivery attempts. Generate STT onset,
interim, end-of-speech, and final events independently of the audio/VAD input timeline.
Do not synthesize all their timestamps from a single “user stopped” action.

Recovery checks use injected errors and approved retry/reset policy to require the
next attempt, continued availability, or closure. Exercise failures on both sides of
a recovery reset and the exact configured tolerance boundary in a long session.
Turn tests enumerate STT-final before/after VAD-end and end-of-speech before/after each,
including shared timestamps. Manual mode has a separate no-automatic-turn-without-
commit contract; explicit host generation is not an automatic turn. Repeated commit,
superseding commit, transcript timeout, and close each need caller-visible checks.

**Content progress and text boundaries.** Provider scripts distinguish transport
keep-alives, empty content, meaningful content, end, and failure. Independently control
first content and subsequent content gaps. Only contract-listed progress observations
reset the corresponding operation/stage timer; keep-alives cannot stand in for content.
Exercise silence before the first token, one token then silence, and sustained content
past the same total duration. The first two require approved inactivity handling; the
last must not fail merely because its total duration is long. This explicitly replaces
any archived expectation that one initial chunk makes an endless silent stream valid.

Use an authored text corpus containing URLs, decimals, abbreviations, Unicode and
punctuation, with expected preservation and any promised segmentation written down.
For each text, vary chunk cuts (including single-character and empty chunks where
allowed) through the provider interface. Compare delivered content to the authored
input and segments to the independently specified policy. Do not run the production
splitter to construct expected segments. Zero-sample audio and sample-rate changes
likewise need actual frame delivery and checks against supported media contracts.

**Running tools and conversation context.** Record the full actual model request
messages, including tool-call and tool-result IDs. Start another request while a tool
body is held, then release or cancel that body and inspect subsequent requests. Check
history consistency against the approved pending-tool representation: completed results
must match their calls, cannot appear twice, and cannot be attributed to a later call.
Whether a pending tool is omitted, represented by a placeholder, or blocks a request is
a product contract to resolve explicitly. A context hash or a call ID alone cannot
verify this behavior. Include late results after a turn or agent replacement.

**Echo routing.** Route copies of actual output audio back through the microphone
boundary with selected delay and gain, mixed with separately supplied user audio.
Record routed frames and the resulting interruption/turn decisions. Use a controllable
speech detector for routing and ownership contracts, but do not claim it validates
acoustic echo discrimination. That claim requires the actual echo-capable media path
and labeled audio fixtures at the adapter/integration layer, with an explicit expected
classification/tolerance contract. If that facility is unavailable, report the acoustic
coverage gap; a fixture-provided “this is echo” label cannot satisfy it.

**Watchdog and reproducible reduction.** A parent process owns the artifact directory
and wall-clock deadline. The child writes and flushes each chosen action before applying
it and records completion positions incrementally. For native inputs, the parent must
retain the raw input before the child executes it. On deadline, the parent terminates
the child and retains the complete input prefix, last completed action, configuration,
and available observations. A deliberately non-yielding child probe must demonstrate
this path; a cooperative async timeout is insufficient. Label this an executor stall,
not an inferred behavioral-rule violation. The watchdog budget is campaign configuration.

Freeze the original failure identity before shrinking: contract ID, violated clause,
and the subject's stable input/request lineage. Candidate deletion retains input IDs;
it does not renumber them into another subject. Replay must reach the original
prerequisite and fail that same obligation. A different failure is saved separately
and does not replace the current candidate. For a watchdog finding, require the same
stalled action/operation lineage and watchdog outcome; inability to reproduce it is
reported as unreproduced. Save originals even when no valid reduction is found.

### Contract coverage and drift checks

Keep a small executable registration table with contract ID, one-sentence guarantee,
input/observation requirements, rule function, deterministic prerequisite probe, and
local counterexamples. Use ordinary Rust data/functions, not a parsed prose catalogue
or a new rule-description language. Split a compound guarantee into independent
clauses before claiming coverage.

The coverage check rejects duplicate IDs, missing required probes/counterexamples,
and a declared supported contract whose execution cannot supply its observations.
An explicitly unsupported clause has a reason and test owner and stays visible as a
gap. Its mere registration does not satisfy implementation coverage. A negative
counterexample must demonstrate the intended rejection, not only exist as a named test.

The table provides structural accountability, not proof the predicate matches its
English guarantee. Review each rule against its stated clause and independent negative
control. Saved cases retain their required obligation and prerequisite; replay fails
coverage validation if the original check becomes unexercised. Do not impose a fixed
historical rule count on the replacement or accept a census of tests as correctness.

Prioritize this infrastructure with the first contracts, not after hundreds of rules
have accumulated. Historical reports of missing negatives and weaker predicates are
reasons to enforce these checks, even where subsequent archive edits changed counts.

### Execution scope

| Layer                      | Required evidence                                                                   | What simulation cannot establish                                         |
| -------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Voice session              | Real session with controlled providers, callbacks, media and clock                  | Actual external transport or binding behavior                            |
| API/binding boundary       | Actual pending handles, caller cancellation, argument conversion and event delivery | A core command test alone cannot prove caller-visible semantics          |
| RTC/provider adapters      | Real adapter with controlled peer/transport, backpressure and disconnects           | An in-memory fake alone cannot prove protocol, queue or adapter behavior |
| Executor/resource boundary | Process watchdog, task/resource observations and explicit scheduler coverage        | Random fake delivery does not enumerate internal poll/thread races       |

Job/process crashes, worker shutdown callbacks, draining, load reporting and worker
memory/resource lifetime belong to worker tests. Voice session teardown remains a voice
contract. Interactions across that boundary require explicit integration ownership;
passing voice simulation must not be reported as worker lifecycle coverage.

Each supported boundary needs a named test owner in its appropriate crate. A skipped
integration layer remains visible. Python reference files and outputs are never loaded
as the behavioral oracle; language-binding integration is a separate boundary check.

### Bounded schedule exploration

Random delays alone are insufficient. Add systematic exploration of permitted
schedules for small bounded cases alongside randomized and coverage-guided scheduling
for larger cases. These modes use the same execution controls and contracts.

| Scheduling dimension            | Required exploration                                                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Independent pending events      | Different delivery orders, constrained only by actual protocol causality                                                       |
| Same virtual timestamp          | Different orders and bursts; equal timestamps do not fix processing order                                                      |
| Overlapping API calls           | Issue later calls before earlier acknowledgements; retain all pending results                                                  |
| Completion/cancellation/closure | Both relative orders and shared-timestamp orderings                                                                            |
| Timeout boundary                | Before, at, and after the configured deadline; include timer delivery among competing events                                   |
| Async tasks                     | Explicitly identify controlled task scheduling points and uncontrolled polls; external event order alone is insufficient       |
| Multiple threads                | Additional execution coverage for real concurrent access; single-thread simulation is not evidence of all thread interleavings |

Before implementing systematic exploration, specify the scheduling mechanism:

- Identify exactly which choices it controls: external delivery, API issuance, timer
  release, task polling, and any thread transitions. Keep unsupported choices explicit.
- Restart each explored branch from a fresh isolated session and controlled dependency
  state, replay its recorded prefix, validate the enabled choices, then select the next
  branch. A different enabled set is divergence, not an equivalent execution.
- Choose how internal scheduling is controlled. If internal polls remain uncontrolled,
  enumerate only the external schedules in that model and report residual nondeterminism.
  Do not claim all internal interleavings merely because external choices were enumerated.
- Validate reset and prefix replay on small cases before adding schedule reduction or
  claiming bounded completion. Persist explored prefixes and unfinished branches.

Use ordinary Tokio for execution, with paused Tokio time for deterministic deadline
control. The session scheduling clock must read Tokio's `Instant` and wait on Tokio
timers; do not combine paused Tokio time with an independently advanced clock.
Wall timestamps remain separate from scheduling deadlines.

Control external delivery through the actual pending fake requests and input channels:
issue calls without awaiting completion, hold or release individual responses/chunks,
and await a specific acknowledgement when a case needs that boundary. Do not use a
fixed number of yields or a short observation timeout as evidence of progress or idle.
Awaiting a timer under paused Tokio lets virtual time advance without a real-time sleep;
it does not assert that every task has settled. Completion assertions await the actual
result, with any behavioral deadline supplied by the contract. A separate process
watchdog detects a stuck test and retains its case.

Tokio chooses internal task polls. Enumerate external delivery schedules only, record
observed outcomes, and report internal poll/thread ordering as uncontrolled. Do not
introduce a bespoke executor or claim exhaustive internal schedule coverage.

Each systematic campaign declares bounds on operations, outstanding work, delivered
events, controllable scheduling choices, and virtual duration. Record schedules
explored, branches left unexplored, and budget termination. Claim exhaustive coverage
only for the declared finite input set and scheduler model when every permitted
branch within those bounds has been explored. Otherwise report partial exploration.

The scheduler chooses among currently enabled events. Enablement reflects real
causality, such as an actual pending request or available frame; it must not suppress
an API call because the runtime is busy or the call might reveal incorrect behavior.
A schedule includes the timing choices within the declared finite domain. Enumerating
orders at one fixed timing does not cover arbitrary timing values.

An event delivery can wake several tasks. Document whether execution settles after
each delivery, releases a burst before polling, or controls intermediate polls; those
are different explored schedules. Timer callbacks and external responses at one
instant must not always receive an implicit fixed priority from the harness.

Do not prune two schedules as equivalent merely because they reach the same visible
state. Any reduction needs a justified independence relation that preserves checked
outcomes, progress obligations, and resource ownership. Keep the original failing
schedule for replay and compare reduced exploration against unreduced small cases.
The implementation mechanism for internal task control and multithread coverage is
still to be selected; neither may be advertised as already exhaustive.

Schedule coverage and assertion coverage are reported separately. Exploring a race
cannot detect its consequence unless a contract checks the affected behavior.

## 6. Progress and recovery checks

Retain real pending calls and record their issue times and actual results. Keep only
the extra per-operation timing needed by progress checks; derive progress from the
contract's specified direct observations. Track acknowledgement, work completion, and
event subscriptions separately without reconstructing a reply lifecycle. Resource
checks read actual creation/release observations. Each contract supplies semantics.

For every operation contract, specify: start event, meaningful progress events, allowed
terminal outcomes, inactivity configuration, timeout handling, and cancellation/close
behavior. A state transition or retry only resets inactivity if the contract explicitly
allows it. Bookkeeping or activity in another operation is not progress. Separately
observe pipeline stages: continued LLM text cannot by itself prove that pending audio
is progressing. Which stages need their own obligations must be specified, not inferred
from a single speech-level heartbeat.

**An injected failure or intentional stall is not automatically a test failure.**
The runtime may correctly time out, retry, reject, cancel, or close according to its
contract. The test fails when that required handling is missing, premature, wrongly
routed, or otherwise incorrect. The checker must independently enforce the deadline;
waiting for the runtime to emit a timeout would miss an absent timeout entirely.

During exploration, check inactivity continuously from request time. Finishing the
random action sequence must not be the first point at which a stall can fail.
After exploration, stop adding work and release finite external blockers under a
bounded policy, including newly created dependencies. Require due work to settle,
then perform cancellation and resource cleanup. Closure must not conceal missing
normal-operation progress. Progressing infinite streams are explicitly ended or
cancelled; one chunk followed by silence is a different case. Releasing dependencies
later cannot erase a deadline violation already observed. Retain dependency-isolation
cases too: releasing everything could conceal an incorrect dependency on an unrelated
callback, such as handoff waiting for enter.

Use virtual deadlines for simulated behavior and an independent process watchdog
for a blocked executor. A timeout reports missing progress within a bound; it does
not prove a lock cycle. A campaign time budget is not a production latency SLA.

Provider retry and session tolerance checks start from known injected provider
outcomes. Match each attempt and its error/retry, rather than compare only totals.
A lost error notification must not eliminate the obligation to account for failure.
Test recovery resets and subsequent failures, not only repeated permanent failure.

Some bindings turn response-channel loss into success, cancellation, or refusal.
Verify the corresponding work/lifecycle evidence; a returned Ok or journalled answer
alone cannot establish correct settlement.

## 7. Readable contracts

A reader should understand a contract in one file without following a multi-scenario
switch or reverse-engineering trace indexes. The contract is independent of the
scenario that happens to exercise it.

### Example: preemptive audio requires authorization

**Contract:** audio belonging to a preemptive reply must not reach playback before
that reply is authorized.

This is the playback-enforcement contract, not proof that authorization itself was
correct. Match playback and authorization to the same reply/generation identity.
Authorization for another reply cannot satisfy the obligation. Specify the actual
playback boundary; synthesized audio in a private buffer is not audible output.
At the same timestamp, timestamp equality alone does not establish causal order.

A separate authorization-permission contract must establish, from independent inputs,
when this reply is allowed to be authorized. Its prerequisite comes from the configured
turn policy and test-controlled user/transcript/turn-decision inputs, not the runtime's
own “authorized” event. Define that predicate explicitly for each supported turn mode
before claiming authorization correctness. Do not invent one universal predicate or
assume the current implementation's decision is the specification.

Together, the checks must detect both early authorization followed by apparently valid
playback and playback that bypasses authorization. The playback contract remains small;
it does not embed the turn detector's policy.

| Shared generator varies                        | Interaction being explored                                             |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| User resumes speaking                          | Speculation becomes stale while generation continues                   |
| Final transcript changes                       | A speculative reply may require replacement                            |
| Multiple speculative generations over time     | Old authorization/results must not be attributed to another generation |
| Host speech, interruption, or handoff          | Other work competes with speculation                                   |
| Provider completion, failure, or late delivery | Results arrive during or after cancellation/replacement                |
| Output detachment or replacement               | Buffered work overlaps output lifecycle changes                        |
| Event order and timing                         | Audio readiness, authorization, and invalidation race                  |

These are examples of dimensions in the shared search space, not branches implemented
inside this rule or an exhaustive list of permitted scenarios. The same contract
checks every applicable generated execution. It does not choose which host calls,
provider events, or schedules the generator may explore.

A deterministic seed establishes an independent condition under which authorization
is forbidden, then lets the provider produce audio. Verify the forbidden condition
from the controlled inputs, audio readiness from the provider, and absence of playback
at the actual output boundary. Also check that no authorization was granted early. This guarantees one meaningful exercise of the boundary;
repeating it with arbitrary delays is not evidence of new behavioral coverage.
Broad generation explores additional operations and orderings around that boundary.

A local negative control supplies playback before matching authorization and must
be rejected. Include a wrong-generation authorization control. The separate permission
checker needs a counterexample where authorization is granted while its independent
prerequisite is false, even if playback follows that authorization correctly. These validate
the checker independently of whether the runtime currently emits the bad sequence.

A separate progress contract requires eligible, authorized work to proceed under its
specified dependencies and cancellation conditions. Permanently suppressing all audio
must not make the suite pass. Whether stale speculation should be cancelled, replaced,
or reused belongs to separate explicit contracts; this rule does not infer those
product decisions from the current implementation.

- One rule, one observable guarantee. Installation, ordering, and completion are separate.
- Keep applicability accounting in shared machinery; report checked/excluded/unreached.
- Helpers find observations by identity; the rule visibly decides correctness.
- Keep a small passing example and a failing counterexample beside each invariant.
- Require missing output from independent input; do not trigger only on emitted output.
- Retain useful repeated setup if it makes the assertion understandable.
- Verify that generated values change real execution and that intended orderings occurred.

## 8. Failures, shrinking, and evidence

Persist concrete operations, configuration, dependency decisions, test identities,
controllable event ordering, and the violated contract before termination where
possible. A process-killed run also needs external artifacts; a Rust panic hook
cannot be its only preservation mechanism.

Shrink while preserving the same violated obligation and necessary prerequisite.
Removing the prerequisite or replacing the failure with another bug is not a valid
reduction of the original finding. Saved replay must assert applicability, not only
absence of a failure. Store decoded input alongside seeds so generator changes do
not silently redefine historical cases. Persist the original failure before shrinking,
with the code/configuration version and a stable contract identity. If replay diverges,
retain the original and classify the result as unreproduced; do not discard it or claim
a smaller case proves the same bug. Record harness failures separately from runtime
contract failures. Inputs reduced to a different defect are separate findings.

Report separately: selected operations, issued operations, observed overlaps,
checked requirements, excluded requirements with reasons, unreached requirements,
failures, and execution failures/skips. Keep fresh generation separate from saved
coverage. Track rare and higher-order interactions without presenting a hit matrix
as exhaustive coverage. Choose weights and campaign budgets from measurements.

For every required deterministic prerequisite/ordering, a missing observation fails its
gate. Broad exploration instead reports measured reachability and uses thresholds chosen
from campaign data; it must not impose an arbitrary requirement that every random run
hit every combination. Unavailable checks never count as passes. Record executions that
reach no behavioral assertion and review why they remain in the search space.

Test the harness as infrastructure: time advances, held dependencies stay held, releases
reach the intended instance, receivers remain observable, and omitted/duplicate/wrong-ID
results are detected. Exercise blocked-executor termination and artifact capture in an
isolated process. An all-green runtime suite cannot independently establish these facts.

Keep local negative controls. The archived suite contains an automated cargo-mutants campaign and four hardcoded
source mutations. Treat their results as historical evidence only. The replacement
strategy must not depend on fragile string replacements or inherit the old campaign
configuration without evaluating it.

### Evaluating reference material

No archived module, runner, generator, registry, or abstraction is required to survive.
Reuse must be justified by clear ownership and demonstrated correctness against explicit
contracts. Inventory actual public boundaries to avoid omissions, but do not infer
expected behavior or test architecture from their implementation locations.

Review archived assertions independently. Retain only guarantees supported by agreed
contracts, replace weak checks, and reject assertions encoding incorrect behavior.
A progressing open stream and a stream that emits one chunk then stalls need distinct
expectations regardless of what old fixtures or tests permit. Archived handoff and
parked-turn expectations likewise need explicit contract review, not automatic adoption.

## 9. Campaigns and implementation acceptance

| Campaign                      | Required output                                                                                                                              |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Per-change checks             | Readable contract/negative tests, saved regressions, boundary cases, and bounded fresh exploration                                           |
| Extended property search      | Multiple seeds, longer histories, higher concurrency, masked and unmasked runs, measured observed coverage                                   |
| Journal-guided search         | Shared semantic checks, persistent cases, journal variants reached per campaign and never produced, failure/timeout status                    |
| Boundary integration          | Results for binding/adapter/executor checks outside voice simulation, with explicit skips                                                    |
| Optional automated mutations  | Clean controls and detected/surviving/invalid mutations reported separately; survivors are coverage evidence, not automatically runtime bugs |

Configure time, memory, and input-size budgets explicitly and report them with tool,
configuration, and code versions. Reaching the exploration budget with progressing work
is not a product timeout: run the declared finish/cancel phase and report incomplete
coverage if it cannot execute. Build failures, missing tools, process kills, and skipped
campaigns are never reported as successful behavioral coverage. Do not confuse replaying
saved fuzz findings with running a new fuzz campaign.

| Order | Deliverable                                                                                                                 | Acceptance evidence                                                                                                                 |
| ----- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1     | API/event/result inventory, contract evidence sources, and scope of scheduling control                                      | Inputs and expected outcomes distinguished; unavailable observations and unresolved policies explicit                               |
| 2     | Minimal nonblocking harness with real pending calls, direct recordings, fresh-session reset and replay                      | Calls/releases proceed while acknowledgements are pending; a recorded concurrent execution replays or reports exact divergence      |
| 3     | Executable output failure/close controls and independent checks, then realtime/parallel tool bodies and STT request scripts | Direct boundary observations satisfy the capability probes; each new contract has a prerequisite probe and rejecting counterexample |
| 4     | Broad/state-aware/native search and bounded schedule enumeration                                                            | Actual overlap measured; scheduler model and unexplored branches reported; original failures retained before reduction              |
| 5     | Obligation-preserving shrinking                                                                                             | Smaller recordings preserve the original violation and prerequisites; a different failure is stored separately                      |
| 6     | Readable contracts across the inventory, with adopted regression cases                                                      | Full test and failure report understandable; required outcomes independently checked; reference assumptions explicitly reviewed     |

Machinery comes first: nonblocking issuance, retained callers, original-failure capture,
replay, independent watchdog artifacts, and contract-registration checks. Then prioritize
output/close behavior, realtime and mixed-outcome tools, and independent STT scripts.
Add each capability to broad generation as it becomes executable. Contract negatives
and drift checks accompany each addition rather than form a final cleanup phase.

A readable contract example can be built alongside shared execution; it must not
replace vocabulary-wide exploration with a series of guessed dangerous pairs.
Consult archived cases as authoring evidence, not as an executable acceptance oracle
for the new design. Prior passing counts do not establish that this plan is implemented.
For each new category, review relevant archived assertions for omitted cases without
treating their catalog as complete. Require observed prerequisites, detection of
missing outcomes, and explicit obligations for any adopted saved regression.
Removing an assertion that contradicts the agreed policy requires a replacement test,
not merely deleting the old failure. Review the actual full test and failure report
for readability before applying its structure to other categories.

## 10. Decisions still needed before dependent implementation

Each of these blocks one family of checkers and nothing else. The machinery they need is built:
the harness can observe every fact listed below, and the generator reaches the states. What is
missing is the product decision, which this document must record before its checker is written.
The implementing agent must not invent one, copy the reference as approval, or mark a dependent
check complete. Each resolution states the allowed outcomes, the progress and deadline policy,
the cancellation behaviour, and one accepted and one rejected example.

| Decision | The question, concretely | Options seen so far |
| --- | --- | --- |
| Event-only waits | `wait_for_interrupted` stays pending after a speech completes normally. Is a waiter who is never interrupted owed an answer, and if so what? | (a) stays pending forever, and the binding must not leak it; (b) resolves "not interrupted" when the speech ends; (c) the subscription is cancelled at speech end and the caller sees a cancellation |
| Meaningful progress per operation | For each work operation, which observations reset its inactivity timer, and what is the limit? Queue time counts from the request, as agreed. | Per leg: a chat delta, a synthesized chunk, a sink push. Whether a keep-alive, a retry, or a state transition counts is undecided |
| Expiry handling | When an operation passes its inactivity limit, what does the driver owe? | (a) a retry within the leg's budget; (b) fail the speech with a provider error; (c) close the session. Today the suite cannot tell "correctly timed out" from "silently stalled" |
| Invalid provider events | Which provider sequences are forbidden rather than merely late, and what is owed for each? | Late delivery after cancellation is normal coverage and stays. Duplicate finals, a final before a start, and a chunk after end are candidates for a separate robustness campaign |
| Internal scheduling | Which task-poll and thread boundaries, if any, become controllable? | Today: external delivery order is enumerated (`runner/schedule.rs`), internal poll order is Tokio's and is reported as uncontrolled. Options are to leave it, or to take a controlled executor |
| Turn-authorization permission | For each turn mode, the independent predicate that says when a reply may be authorized. Without it, `playback_requires_authorization` checks only that playback follows authorization, not that authorization was permitted. | Needs one predicate per mode from the configured turn policy and the case's own user/transcript inputs, never from the runtime's own `Authorized` event |
| Competing handoffs and tool continuation | When two parallel tool calls both request a handoff, which wins and what happens to the loser? When may a parallel tool result start a follow-up request, and separately for server-driven and client-driven continuation? | Never "whichever the current implementation produced". The allowed winners and refusals must be named first |
| Pending-tool history | A new request arrives while a tool is still running. Does the request wait, omit the pending call, or carry a defined placeholder, and how does the eventual result update history? | The suite records the full actual request messages and tool-call ids, so any of the three is checkable once chosen |
| Output shutdown | Per output, the graceful-close obligation: which operations must complete, under what deadline, and when forced cleanup is allowed. | `provider_streams_closed` covers provider streams only. Output close is unchecked, and `output_replacement_is_clean` is the ownership half, not the shutdown half |
| Acoustic echo | Whether echo discrimination is in scope, and if so at which layer with which labelled fixtures. | A content-blind detector cannot establish it. Either an adapter-layer test with labelled audio and a stated tolerance, or the gap is declared and reported |

Two of these are worth deciding first, because the campaign already produces the states and the
contracts are the ones most likely to catch a real defect: meaningful progress per operation
(which turns "a speech never finished" from a fixture-budget observation into a product contract)
and turn-authorization permission (which completes the only speculative-playback contract we
have). The other eight can wait without blocking anything already built.

Inventory, nonblocking issuance, identity tracking, artifact capture, checker validation, and
everything in §9 that does not name one of these semantics proceed independently, and have.
