# Testing, evals and simulations

Use programmatic tests for specific agent behavior, conversation judges for
transcript-based evaluation, and simulations for complete scenarios. All three
remain supported. Simulations are the main experience for scenario execution
and judging; the SDK also supports tests and evaluations without a simulation.

| Surface | Use it for | Evidence |
| --- | --- | --- |
| Run event assertions (`result.expect`) | A particular message, tool call, argument or event sequence | Events of one `agent.run()` invocation |
| Conversation evals (`livekit.agents.evals`) | Scoring a supplied conversation with custom or built-in judges | Transcript, optional reference and individual judgments |
| Simulations and `SimulationContext` | An entire scenario with a simulated user and application checks | Simulator verdict plus application-owned verification |

Agent execution and result types are defined in [agents.md](agents.md).
Simulation callbacks belong to the [server entrypoint](server.md) lifecycle.

## Programmatic tests

Keep small tests close to application code. Run the agent with `agent.run(...)`
and assert its typed result and the application's state directly. Use run
event assertions when the behavior under test is which messages or tools the
agent produces:

```python
result = await agent.run("Book a room for tomorrow.", context=app)

result.expect.next_event().is_function_call(
    name="check_availability",
    arguments={"room_type": "king"},
)
result.expect.next_event().is_function_call_output()
result.expect.next_event().is_message(role="assistant")
result.expect.no_more_events()
```

`result.expect` is on every [`RunResult`](agents.md#running-a-turn), from
`agent.run()` and from a `TextSession` turn's `turn.result`. Its events are this
invocation's assistant messages, tool calls and tool outputs, in order: the
items of `result.new_messages()` without the incoming message. Plain list
assertions over `result.new_messages()` work as well. `agent` and its model are
test fixtures that produce the expected exchange. Argument assertions check
the supplied keys; assert exact business state separately when extra fields or
effects matter. `next_event` advances through the observed sequence.
`contains_function_call` and `contains_message` check for an event without
requiring that exact sequence. Failed expectations raise `AssertionError` with
event context.

| Assertion | What |
| --- | --- |
| `result.expect[i]`, `result.expect[i:j]` | the event at an index, or a range of events with the `contains_*` checks |
| `next_event(type=...)` | the next event, narrowed to `"message"`, `"function_call"` or `"function_call_output"` |
| `skip_next(count=1)` | advance past events without asserting them |
| `skip_next_event_if(type=..., ...)` | skip the next event only if it matches; returns its assertion or `None` |
| `contains_function_call`, `contains_function_call_output`, `contains_message` | an event anywhere in the run or range |
| `no_more_events()` | nothing left after the current position |

Each assertion's `.event()` returns the matched `RunEvent`: a
`ChatMessageEvent`, `FunctionCallEvent` or `FunctionCallOutputEvent`, each
with its `item` and literal `type`.

A message assertion can use an LLM judge for semantic intent:

```python
await result.expect.contains_message(role="assistant").judge(
    judge_model,
    intent="Explains that the requested room is unavailable.",
)
```

`judge_model` is an application-supplied LLM. This check performs a model call;
ordinary event assertions do not. A test using a live agent model or judge
requires its provider even though no simulation is running. Use controlled
models and service fixtures for hermetic tests.

`mock_tools(agent, {"tool_name": replacement})` from `livekit.agents` scopes
replacements to invocations started inside its synchronous context manager.
It takes the constructed agent, and a run retains its selected mocks until it
settles. Concurrent runs outside the block keep their own tool implementations.

## Conversation evals

`livekit.agents.evals` evaluates an existing `ChatContext`. It does not create
a simulated user or execute the conversation. It works in tests, application
evaluation jobs and session-end callbacks.

```python
from livekit.agents.evals import JudgeGroup, accuracy_judge, tool_use_judge

judges = JudgeGroup(
    llm=judge_model,
    judges=[accuracy_judge(), tool_use_judge()],
)
evaluation = await judges.evaluate(chat_ctx, reference=reference_chat_ctx)

assert {"accuracy", "tool_use"} <= evaluation.judgments.keys()
assert evaluation.all_passed
```

`chat_ctx`, the optional reference conversation, and `judge_model` are supplied
by the application. `JudgeGroup` accepts an LLM instance or an inference model
identifier and runs its configured evaluators concurrently. When invoked in a
job context it also records judgment results as session tags.

The public evaluation types are:

| Type | Contract |
| --- | --- |
| `Evaluator` | A named evaluator with `evaluate(chat_ctx=, reference=, llm=)` returning a `JudgmentResult` |
| `Judge` | Base class for application-defined evaluators; override `evaluate` for programmatic or custom checks |
| `JudgmentResult` | `verdict` (`pass`, `fail`, `maybe`), explanatory `reasoning`, evaluation `instructions`, and the predicates `passed`, `failed` and `uncertain` |
| `JudgeGroup` | Evaluate one conversation with a collection of evaluators |
| `EvaluationResult` | Named `judgments`, aggregate `score`, and predicates such as `all_passed`, `any_passed`, `majority_passed` and `none_failed` |

Built-in factories cover `task_completion_judge`, `handoff_judge`,
`accuracy_judge`, `tool_use_judge`, `safety_judge`, `relevancy_judge`,
`coherence_judge` and `conciseness_judge`. Applications choose the criteria
appropriate to the conversation; a custom `Judge` can check data without an
LLM. Reference conversations are optional and interpreted by each evaluator.

The aggregate score gives `pass` a value of 1, `maybe` 0.5 and `fail` 0.
`all_passed` requires passing judgments; `none_failed` also accepts uncertain
judgments. A judge that raises, or whose model or evaluator gives no `pass`,
`fail` or `maybe` verdict, is logged and absent from `judgments`. Check that all
requested judgments are present before interpreting the aggregate: a missing
judgment or an empty group is not evidence of success.

A transcript judge cannot establish that an external write actually committed.
Verify bookings, payments and other effects against application records.

## Simulations

Simulations drive a scenario through the application with a simulated user.
The simulator evaluates the conversation; application code can add checks
against its services and records. Scenario execution and result presentation
belong to the simulations product rather than a second SDK scenario runner.

The SDK supplies a `SimulationContext` during a simulation. Access it through
`JobContext.simulation_context`, or `current_simulation()` from
`livekit.agents.job`; both return `None` outside a simulation. Register an
`on_simulation_end` callback on the voice entrypoint to inspect the final
verdict and verify application outcomes.

```python
from livekit.agents import AgentServer, JobContext
from livekit.agents.simulation import SimulationContext

server = AgentServer()

async def check_booking(ctx: SimulationContext) -> None:
    expected = ctx.userdata()["expected_booking"]
    actual = await bookings.for_simulation(ctx.simulation_job_id)
    if actual != expected:
        ctx.fail("Saved booking does not match the scenario's expected booking.")

@server.rtc_session(on_simulation_end=check_booking)
async def phone(ctx: JobContext) -> None:
    # Start the application's VoiceSession as usual.
    ...
```

`bookings` is an application-owned test service, keyed here by the simulation
job id, and `expected_booking` is application-defined scenario data. Use
isolated test services for scenario effects. The callback can also run SDK
judges when additional transcript criteria are useful.

### SimulationContext

| Member | Meaning |
| --- | --- |
| `scenario` | The scenario being exercised |
| `userdata()` | Scenario-provided JSON application data, or an empty object |
| `simulation_mode` | Text or audio interaction mode; unspecified is treated as text |
| `simulation_run_id` | The overall simulation run id, available from entrypoint execution |
| `simulation_job_id` | The individual scenario job id, available from entrypoint execution |
| `job_context` | The running job, including its room and primary session |
| `simulator_verdict` | Read-only `SimulationVerdict(success, reason)`, available in `on_simulation_end`; earlier access raises |
| `fail(reason)` | Record an application failure that vetoes a simulator pass |
| `user_verdict` | The application's failure verdict, or `None` if no veto was recorded |

The framework records the simulator and application verdicts separately.
The effective result passes only when the simulator passes and the application
has not vetoed it. There is no application `success()` override: application
checks cannot rescue a simulator failure. Repeated `fail` calls retain the
last reason. Returning without calling `fail` leaves the simulator verdict
unchanged. Scenario userdata is an application payload, separate from persisted
agent context.

## Implementation ownership

The shared agent core owns deterministic run/event assertions. Evaluation
helpers follow the owning subject's pure-algorithm rules; model calls and judge
orchestration are async Rust in `lk-agents-inference`, and application callbacks
run through HostInvoker.
The SDK judge API remains supported without requiring a separate scenario or
judging subsystem inside the agent core. Simulation verdicts and SDK judgments
keep their distinct result types and meanings.

Programmatic tests need no room or session: `agent.run()` runs the agent
directly. The framework ships no test-only room or fake export; its own
room-level behavior is tested in Rust over simulated SDK peers.

The framework's deterministic simulation runs the real voice, room and avatar
executors and the real decision cores, with fakes only at the SDK and provider
ports. Executors spawn and sleep only through an injected spawner and clock, so
a deterministic scheduler using basic partial-order sampling decides every
interleaving, and every `select!` is biased so no branch is chosen at random.
The [Rust plan](internals/rust-core.md#deterministic-simulation-testing) owns
that contract.

The [Rust plan](internals/rust-core.md#support-algorithms-and-observations)
defines these implementation boundaries; this page owns the public testing,
evaluation and simulation-context contracts.
