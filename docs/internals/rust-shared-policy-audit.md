# Shared policy and coordination audit

Use `AgentRun` as the single owner of agent-invocation decisions. Voice, TextSession, runtime,
and host adapters supply their domain facts and execute requests; they should not independently
reinterpret the same raw outcomes. This audit extends the [transition plan](rust-transition-inventory.md)
and [lifecycle naming audit](rust-lifecycle-symmetry-audit.md).

This is a read-only source audit of Rust core/voice/runtime/inference/net/storage/worker and
Python host integration, with focused tracing of agent/tool outcomes, definition projection,
request planning, retry composition, and cancellation. It is not an exhaustive semantic proof
of every crate or provider. The initial source audit ran no new runtime tests; the implementation checkpoint below records
the subsequent fixes and verification. Findings distinguish
observed code differences from suspected failures and intentional compatibility differences.
The shared-run cleanup and naming corrections are implemented. Question-selection behavior
is under the explicit scope correction below; broader transition-plan work remains separate. The observation inventory preserves the pre-correction evidence for review.

## Question policy is outside the generic invocation loop

**User decision after reviewing PR #8256:** finish the Rust architecture cleanup first, then
integrate the delegation/question work as a separate behavior change. Do not implement a
competing question queue or finalize multi-question selection while that contract is pending.
The previous “first terminal question wins” description was too broad and is not the accepted
conversation policy.

`AgentRun` is the shared generic invocation loop in this codebase, not a separate voice-only
`AgentLoop`. It owns model/tool execution, approvals, retry/output validation, canonical
outcomes, and cancellation accounting. It must not decide which pending human questions to
discard, mark an unselected question as answered, prioritize experts, or schedule what the
person hears. A singular `RunResult.question` field does not establish permission to lose
other correlated input requests. The exact generic result/input-request representation is to
be reconciled with delegation rather than patched with another core-owned queue.

| Responsibility | Owner |
| --- | --- |
| Execute the agent and report correlated tool outcomes / input requirements | Generic `AgentRun` |
| Preserve expert task/request identity, resume the correct backend, retain durable routing state where required | Delegation and backend conversation/lane routing |
| Keep pending questions, choose a foreground request, ask through the fast brain, and route that answer | Live conversation / `VoiceSession` frontend inbox |
| Carry messages, perform host calls, and execute effects | Runtime and transport adapters |

Current code needs this distinction stated explicitly:

- Live `RemoteAgent` detaches while the expert works. Its input request is delivered as a
  session update through `_ask_the_person`, not as a terminal `ToolExecution::Ask` in the
  original frontend run. The existing `Experts` inbox preserves different experts' requests
  and presents one at a time. Answering one makes the next eligible; it does not discard it.
- The recent first-only regression concerns direct local `InputRequired` results in one
  invocation. Shared `step_outcome.rs` still selects one and changes later question outputs
  into plain text. That existing core policy and the new test that pins it are **provisional**
  and must be revisited. Passing that test is not acceptance of losing other pending requests.
- Multiple legitimate final output candidates and multiple human input requests are different
  contracts. Moving question orchestration out does not move approval, retry-budget, or output
  validation policy into voice or reintroduce voice `StepDecision`.

Source: [live remote adapter](../../livekit-agents-next/livekit/agents/a2a/remote.py),
[expert update dispatch](../../rust/crates/lk-agents-voice/src/agent_activity/experts.rs),
[frontend inbox](../../rust/crates/lk-agents-voice/src/experts.rs), and
[sequential-question test](../../rust/crates/lk-agents-voice/src/experts_tests.rs).

### Integration with PR #8256

Reviewed the PR description and diff at head `317f826957239e255dd67dd162c83b4cac6b072b`:
[voice: the call on its lane, and a text during a call](https://github.com/livekit/agents/pull/8256).
Its Python activity retains questions, presents one foreground question, correlates the answer,
and can prioritize a follow-up from the expert task holding the floor. Its regression expects
“Which date?” → “Morning or evening?” → “Which seat?”. The PR also connects pending questions
across text and calls through lane storage. The description explicitly leaves relay/claim/
read-back lifetime redesign as a TODO and says the PR contains no Rust change; do not describe
those remaining choices as fully settled by the current patch.

After the agreed architecture work, reconcile that PR with the existing Rust expert inbox and
feature contracts as one question/delegation implementation. Do not blindly cherry-pick its
Python scheduling machinery into sans-I/O core, or duplicate persistent question state beside
the existing owner. Resolve any behavior differences in the shared contract, then adapt the
runtime/transport ports and tests together.

Integration acceptance must cover both questions being retained, one foreground question at a
time, each answer reaching only its own request, interrupted/unheard questions remaining
pending, stale/earlier human turns not answering a later question, handoff/task floor ownership,
and cross-text/call continuation. Revisit the generic same-invocation multi-`InputRequired`
representation with that work. Before freezing the architecture interface, define a lossless
reporting seam; defer question queues, priority, and presentation to the separate integration.
This section records a deferred design correction; it does not claim the working code has
already removed that assumption or that PR #8256 has been integrated.

## Historical names: StepSettled and StepVerdict

Before the rename, the coordinated handshake was narrower than its names suggested:

1. Tools belonging to a response have settled. `AgentRun` computes canonical recorded items,
   retry/failure/output precedence, terminal selection, and whether it would continue.
2. `AgentRunEffect::StepSettled { step, items, continues }` asks the voice coordinator to handle
   that boundary before the invocation proceeds. It does not mean the run or speech ended.
3. Voice handles its history/publication, host tool-executed callback, and floor/interruption
   constraints, then sends `AgentRunEvent::StepVerdict` with a correlated step ID.
4. `AgentRun` applies the permitted restriction. Failure and accepted output retain their
   precedence. Voice does not gain authority to select a different terminal tool outcome.

Source: [shared outcome calculation](../../rust/crates/lk-agents-core/src/agent_run/step_outcome.rs),
[coordinated constraints](../../rust/crates/lk-agents-core/src/agent_run/coordinated.rs), and
[current voice adapter](../../rust/crates/lk-agents-voice/src/agent_activity/reply_run.rs).

The handshake is for tool-result processing, not a universal notification after every model
response. A response without calls follows `text_output` directly. Conversely, an answer
routed straight to a tool can enter this boundary without a preceding model request. These
facts make `ModelResponseFinished` and a generic “every step completed” name misleading.

## SDK comparison

Checked the current official documentation on 2026-11-01. These are different execution
architectures; none is a drop-in voice coordination protocol.

| Project | Existing terminology and model | Lesson for this codebase |
| --- | --- | --- |
| OpenAI Agents Python SDK | Internal `SingleStepResult.next_step` selects `NextStepRunAgain`, `NextStepFinalOutput`, `NextStepHandoff`, or `NextStepInterruption`. The SDK marks these internal types outside its public surface. | Prefer a canonical run-loop outcome and concrete action names. Do not duplicate terminal selection in voice. |
| PydanticAI | Agent execution uses graph nodes such as `UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, and `End`, carrying a final result. Iteration advances the same run and history. | Describe the operation being performed. Do not introduce a new graph/node framework just to borrow the vocabulary. |

Sources: [OpenAI SDK internal run steps](https://openai.github.io/openai-agents-python/ref/run_internal/run_steps/),
[OpenAI agent-loop guide](https://developers.openai.com/api/docs/guides/agents/running-agents), and
[PydanticAI agent iteration](https://pydantic.dev/docs/ai/core-concepts/agent/#iterating-over-an-agents-graph).

The recommendation below is our design inference, not terminology these SDKs use for an
identical handshake. Their ordinary next-action language is useful; copying the exact internal
type count or exposing their node machinery would add concepts without solving our boundary.

## Adopted replacement names

These names are implemented with the shared ownership change. Voice `StepDecision` is
removed rather than renamed. The replacement names describe the existing coordination
boundary; they do not add another decision layer.

| Former name | Implemented replacement | Meaning |
| --- | --- | --- |
| `AgentRunEffect::StepSettled` | `AgentRunEffect::HandleToolResults` | Requests coordinator processing of the canonical tool-result boundary |
| `AgentRunEvent::StepVerdict` | `AgentRunEvent::ToolResultsHandled` | Reports processing finished and supplies permitted continuation constraints |
| Core `StepVerdict` type | `AgentRunContinuation` | Existing continue/stop/interruption options, scoped to the owning invocation |
| `StepVerdict::Continue { tools_forbidden }` | `AgentRunContinuation::Continue { tools_forbidden }` | Proceed with the shared run's chosen outcome; restrict tools if it requests another model step |
| `StepVerdict::End` | `AgentRunContinuation::Stop` | Suppress another model step if the shared run would otherwise continue |
| `StepVerdict::Interrupted` | `AgentRunContinuation::Interrupted` | Apply the shared interruption rule, preserving failure/accepted-output precedence |
| Voice `StepDecision`, proposed `ToolStepDecision` | Remove | No second outcome-selection or continuation-policy implementation |

Current implementation shape, reusing existing IDs and enum boundaries. The names are adopted;
the input-request payload is not frozen. In particular, `terminal_call` must not become a
contract that discards other human input requests:

```rust
// AgentRun -> voice coordinator: canonical items, not raw outcomes to decide again.
AgentRunEffect::HandleToolResults { step, items, continues, terminal_call }

// Voice coordinator -> same AgentRun, after its required processing.
AgentRunEvent::ToolResultsHandled {
    step,
    continuation: AgentRunContinuation::Continue { tools_forbidden },
    reply_required, // permitted host edits, keyed by call ID
}
```

`Continue` cannot force a retry after terminal failure, undo accepted output, or choose a
new question. `Stop` is not cancellation of all voice work. None implies playout completion.
Preserve step correlation and reject stale replies through the existing run phase/ID checks.
No additional result wrapper, execution phase, queue, or public SDK primitive is introduced.
Use the existing types directly; a separate `ToolResults` record is not required by this plan.

The implemented handshake accepts only per-call `reply_required` edits. Core applies them
to its canonical history and recomputes ordinary continuation when permitted; explicit
terminal/failure/accepted-output selection remains fixed. Voice applies the same edits to
the supplied canonical items for its published record. Unknown/stale step IDs cannot patch
another step. No arbitrary replacement history is accepted.

## Implementation checkpoint

- **DP01–DP03 cleanup:** shared core owns canonical invocation outcomes, history construction,
  and host-edit continuation. The multi-question selection policy is not accepted; see the
  question-owner correction above. Voice consumes canonical items; its current selected-question
  payload remains provisional. The duplicate
  `StepDecision`, `StepInputs`, `NextStep`, and `ReplyRequest` policy wrappers are removed.
  Completion re-reads current interruption state after host callbacks, so a delayed callback
  cannot resurrect a question after the person interrupted.
- **Canonical history clarification:** both paths now use calls-then-outputs, each in model
  order. This matches the existing shared effect's documented contract and preserves actual
  call-dispatch and output-settlement timestamps. The initial suggestion to preserve
  interleaved call/output pairs overlooked voice's timestamp-ordered storage; simply changing
  its commit loop would not preserve that order in the stored history or next model request.
  Core now constructs the agreed order once; voice does not reconstruct it independently.
- **DP04:** shared duplicate policy receives validated live-execution facts from existing voice
  records, including detached bodies and concurrent replies. Only admitted/executing bodies
  count, not argument preparation or pending approval. Cross-reply replacement targets the
  stable execution operation, not a potentially reused model call ID. No second live registry.
- **DP05:** advertised functions must retain installed policy metadata. Missing metadata is
  refused at startup and diagnosed at reply construction instead of silently disabling
  approval/duplicate/retry policy. Custom offered schemas can still project canonical policy.
- **DP06:** direct/text and voice output conversion share one ordinary runtime helper, retaining
  their distinct serialized-result and live host-value lifetimes.
- **DP07:** reply construction extracts its initial plan and processes every remaining startup
  effect through the normal owner. It no longer drops them through `find_map`.
- **DP08:** core cleanup deadlines participate in voice wakeups. Validation cancellation targets
  the exact host operation/task. Speech-owned validation/refusal callbacks settle on cancellation
  or release, while detached bodies and deferred tool-reply rendering keep their own lifetime.
  Core request cancellation uses forced producer interruption with its normal stream-close path
  and watchdog. A failed text validator no longer waits for its own already-received answer.
- The DST spawner retains its deterministic Tokio handle so teardown outside an entered runtime
  can schedule cleanup on that same executor. This fixes the missing-reactor test obstruction.

The retained/compatibility entries DP09–DP17 remain deliberate distinctions, not pending requests
to merge their types. Tests exercise shared terminal precedence, reverse completion order,
canonical stored/model history, host edits, detached duplicates, unknown metadata, exact callback
cancellation, and deadline-driven cleanup. See the validation results recorded with this change;
passing baseline tests alone are not evidence for a different, unimplemented architecture.
The current deadline-before-event dispatch order also needs an explicit tie contract and
focused before/equal/after-deadline tests before the coordinator refactor.

### Verification of the implemented corrections

- Core: 426 library and 5 integration tests passed.
- Voice: 172 library and 7 integration tests passed.
- Runtime: all 174 library tests passed. Seven local-socket tests initially hit sandbox bind
  restrictions; the full rerun with local socket access passed.
- Fuzz library/oracle contracts: 25 tests passed, including the interruption-provenance
  checker counterexamples. Voice/room deterministic regression target: 208 tests passed,
  including the terminal-selection,
  canonical model-history, detached-duplicate, VoiceTask, and deferred-reply cases.
- Architecture checks passed with zero warnings in the 15 warning-clean crates; no exemption
  or baseline was broadened. Architecture checker counterexamples (47) and codegen tests (132)
  passed. These internal Rust changes introduce no new host export declarations.

Relevant source: [shared coordinated tests](../../rust/crates/lk-agents-core/src/agent_run_tests.rs),
[duplicate policy tests](../../rust/crates/lk-agents-core/src/tool_calls_tests.rs),
[voice policy/deadline tests](../../rust/crates/lk-agents-voice/src/agent_activity/reply_run_tests.rs),
[delayed-hook interruption](../../rust/crates/lk-agents-voice/src/agent_activity/tool_results_tests.rs),
[runtime callback ownership](../../rust/crates/lk-agents-runtime/src/voice/effects_tests.rs), and
[simulation regressions](../../rust/crates/lk-agents-fuzz/src/tools_tests.rs).

### Separate contract still to decide

A VoiceTask ends through its validated `Completed`/`Failed` tool transition and restores the
parent floor; this is intentionally distinct from the shared invocation's accepted output.
A batch mixing task completion with another tool's `InputRequired` needs an explicit terminal
precedence rule before that overlay is redesigned. Do not claim the shared invocation fix
resolves every mixed VoiceTask batch, or silently change task-result validation/context to
make the types look identical. Add the mixed-batch regression with the eventual contract.

## Discrepancy inventory

“Confirmed difference” means the code paths visibly differ; it does not claim every difference
has a reproduced user-visible failure. “Risk” requires focused runtime regression coverage.
This table preserves **pre-correction observations**, not a fresh list of pending fixes.
The implementation checkpoint records DP02–DP08 corrections and the shared-owner part of
DP01. Human-question cardinality in DP01 remains provisional. DP16 hook grouping is implemented;
its owner-derived run-completion redesign remains planned.
The [main plan status table](rust-transition-inventory.md#plan-audit-checkpoint) owns sequencing.

| ID | Priority / confidence | Finding and evidence | Direction and verification |
| --- | --- | --- | --- |
| DP01 | High; confirmed competing selection | Core chooses the first terminal outcome in model order. Voice collects every delivery; multiple `InputRequired` deliveries can leave the last question/tool in voice state. [Core](../../rust/crates/lk-agents-core/src/agent_run/step_outcome.rs), [voice selection](../../rust/crates/lk-agents-voice/src/tool_executor.rs), [delivery](../../rust/crates/lk-agents-voice/src/agent_activity/tools.rs) | Keep shared final-output/failure precedence. Human questions require lossless correlated reporting before API freeze; selection/presentation belongs to the later delegation integration. The current first-only regression characterizes provisional code, not an accepted contract. |
| DP02 | High; confirmed representation difference | Core records call/output pairs in model order; voice `commit_of` concatenates all calls then all outputs. The adapter ignores canonical `StepSettled.items`. [Core](../../rust/crates/lk-agents-core/src/agent_run/step_outcome.rs), [voice commit builder](../../rust/crates/lk-agents-voice/src/tool_executor.rs), [adapter](../../rust/crates/lk-agents-voice/src/agent_activity/reply_run.rs) | Use canonical domain items/order. Provider-specific formatting may project later; do not independently rebuild domain history. Test multiple calls, persisted history, follow-up model input, and realtime push order. |
| DP03 | High; confirmed split update paths | `function_tools_executed` edits change voice `reply_required` and rebuilt history/continuation. Core has already extended its history, while `StepVerdict` carries no edited outputs. [Voice hooks/commit](../../rust/crates/lk-agents-voice/src/agent_activity/tools.rs), [shared history](../../rust/crates/lk-agents-core/src/agent_run/step_outcome.rs), [reply contract](../../rust/crates/lk-agents-core/src/agent_run/coordinated.rs) | Define edits/acceptance in the shared coordinated contract. Test core, voice, persisted, and realtime views after cancellation-of-reply edits; no independent second history policy. |
| DP04 | High; source-level liveness/policy risk | Core duplicate checking scans its current `Calls` in `Phase::Running`. First voice `ctx.update()` settles that entry while the body stays live in voice. Later steps/replies can lose that running duplicate from the policy's view. [Calls](../../rust/crates/lk-agents-core/src/tool_calls.rs), [voice running bodies](../../rust/crates/lk-agents-voice/src/agent_activity/tools.rs), [call-wide contract](../tools.md#on-a-call) | Feed session-wide still-running execution facts into one shared duplicate-policy owner. Test reject/replace/confirm across detached updates, later steps, concurrent replies, name/argument scope, and final release. Do not add a second voice duplicate check. |
| DP05 | Medium/high; observed permissive default | Voice reconstructs `AgentDefinition` by tool name. Unmatched advertised specs default to no approval, duplicate allow, default scope/retries, and no remote metadata. [Voice projection](../../rust/crates/lk-agents-voice/src/agent_activity/reply_run.rs), [Python definition](../../livekit-agents-next/livekit/agents/voice/agent_run.py) | Distinguish intended custom/session tools from lost metadata. Project canonical `FunctionToolInfo` where available; specify unknown-tool behavior. Test known-policy preservation and unexpected mismatches. No claim all unmatched tools are bugs. |
| DP06 | Medium; duplicate boundary conversion | Direct runtime and voice separately map `ValidatedOutput` to core verdict plus host value. Voice does not need the direct/text serialized result in the same way. [Direct conversion](../../rust/crates/lk-agents-runtime/src/agent_run/execution.rs), [voice conversion](../../rust/crates/lk-agents-runtime/src/voice/effects.rs) | Share an ordinary conversion helper if it reduces code. Preserve distinct live host-value versus serializable-result ownership. Parity cases: accepted, retry, failed, and host error. No new validation coordinator or enum. |
| DP07 | Medium/high; observed discarded outputs | Reply startup selects one `PlanRequest` with `find_map`, discarding other startup effects. [Adapter startup](../../rust/crates/lk-agents-voice/src/agent_activity/reply_run.rs) | Process all effects or explicitly enforce the permitted startup contract. An unexpected resource/cancel/observation effect must not vanish silently. |
| DP08 | Medium/high; cleanup ownership risk | Voice ignores several core cancellation/release effects, and speech release removes the synchronous run without necessarily sending `Cancel` or advancing its cleanup deadlines. [Adapter](../../rust/crates/lk-agents-voice/src/agent_activity/reply_run.rs), [shared cancellation](../../rust/crates/lk-agents-core/src/agent_run/mod.rs) | Map every ignored effect to a concrete alternative owner. Test interruption/close during argument preparation, validation, and requests. Dropping a state machine is not evidence its I/O settled; source inspection does not prove every ignored effect leaks. |
| DP09 | Retain; different result context | VoiceTask `validate_tool_completion` validates a tool transition with its tool RunContext. Agent invocation `validate_output` validates text/structured candidates and serialization with a different context. [Host validation](../../livekit-agents-next/livekit/agents/voice/agent_host.py), [voice tool execution](../../rust/crates/lk-agents-runtime/src/voice/generation/tools.rs) | Document and test the two contracts, including validation error/ModelRetry/ToolError behavior. Share only genuinely equivalent conversion code; do not merge away VoiceTask context or invoke validators twice. |
| DP10 | Retain; existing reuse | Outcome shape, verbatim delivery, approval marks, and retry counting already use shared core helpers. [Voice tool adapter](../../rust/crates/lk-agents-voice/src/tool_executor.rs), [shared outcome](../../rust/crates/lk-agents-core/src/tool_outcome.rs) | Keep reuse. Remove duplicate wrapper decisions around it rather than create another shared layer over existing helpers. |
| DP11 | Retain; existing shared policy | Provider/HTTP/fallback paths already compose `core::attempts::Attempts`; adapters observe different transport/protocol phases. [Provider](../../rust/crates/lk-agents-inference/src/provider.rs), [HTTP](../../rust/crates/lk-agents-net/src/http/exchange.rs), [fallback](../../rust/crates/lk-agents-inference/src/fallback/machine.rs) | Keep the common attempt engine and boundary-specific facts. Do not merge distinct protocols into one large retry machine. |
| DP12 | Retain; intentional retry composition | Fallback and gateway adapters disable inner retries where an outer owner already retries. [Fallback](../../rust/crates/lk-agents-inference/src/fallback/mod.rs), [gateway](../../rust/crates/lk-agents-inference/src/llm/gateway/mod.rs), [Python fallback](../../livekit-agents-next/livekit/agents/llm/fallback_adapter.py) | Document one retry owner per composition; test that attempt counts do not multiply. A retry field in two layers is not proof of competing policy. |
| DP13 | Intentional compatibility discrepancy | Public Python `APIStatusError(499)` can be retryable; native Rust explicitly normalizes 499 to cancelled/non-retryable. Both document the distinction. [Python](../../livekit-agents-next/livekit/agents/_exceptions.py), [native taxonomy](../../rust/crates/lk-agents-abi/src/errors/api.rs) | Keep an explicit constructor-to-native mapping and parity tests. Changing public compatibility requires a contract decision; do not silently normalize both constructors. |
| DP14 | Retain; protocol-specific completion | Host STT attempt handling and native STT differ because native finalization sees provider acknowledgment/quiet-period semantics. [Host-facing attempts](../../rust/crates/lk-agents-inference/src/provider.rs), [native lifecycle](../../rust/crates/lk-agents-inference/src/stt/lifecycle.rs) | Keep protocol finalization distinct while reusing attempt policy. Input EOF, provider EOF, and complete transcription are different facts. |
| DP15 | Retain; one planning helper, distinct snapshots | Core request planning is shared; voice supplies session history, instruction/floor state, pending items, and realtime projections. [Shared planner](../../rust/crates/lk-agents-core/src/request.rs), [voice planning](../../rust/crates/lk-agents-voice/src/agent_activity/reply_run.rs) | Keep snapshot ownership explicit and test the chosen history. Do not classify all voice planning as a second loop or duplicate planner. |
| DP16 | Selected simplification; lifecycle overlap | Hooks already share terminal handling, but exit adds another notification layer; run work can also duplicate existing owner liveness. [Symmetry audit](rust-lifecycle-symmetry-audit.md), [run completion plan](rust-transition-inventory.md#voice-run-completion) | Reuse common completion with deferred ordering; prefer owner-derived run completion. Do not add a second registry merely to represent queued completions. |
| DP17 | Retain; shared invocation plus other domains | TextSession composes AgentRun. A2A projects protocol outcomes; voice speech/room completion includes playout and session work. [TextSession](../../rust/crates/lk-agents-core/src/text_session/mod.rs), [A2A](../../rust/crates/lk-agents-core/src/a2a/task/mod.rs), [voice result](../../rust/crates/lk-agents-voice/src/run_result.rs) | Preserve distinct conversation, transport, invocation, and speech lifetimes. Reuse invocation policy without turning it into a global session/transport state machine. |

## Implementation gate

Preserve the implemented canonical history/edit semantics, session-wide duplicate visibility,
and removal of voice decision wrappers. Do not schedule these corrections a second time.
Before the remaining architecture changes, resolve the lossless input-request seam, run
membership/result contract, and deadline/event ordering listed in the main plan. Keep
transport/presentation projections explicit. Integrate PR #8256 separately after that cleanup;
no question queue or priority policy is authorized by the shared invocation refactor.

For each alleged duplicate, answer: which owner decides, which facts it receives, which
result another layer consumes, and whether that layer is projecting or choosing again. Share
an ordinary helper for equivalent conversion; retain a distinct owner for different lifecycle
facts. Add no general policy framework, aliases, or extra decision enum solely for this audit.
