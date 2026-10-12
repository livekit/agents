# Rust core: planned

Not built yet. This extends [rust-core.md](rust-core.md).

## The run unit of work in Rust

The run unit of work and the turn belong to `lk-agents-storage`, not to
Python: a Node package gets the same mechanism, and its tests are the
crate's, run once against a real agent-db. Python and Node keep the
developer surface and the three things only a host can do: run the handler
(call the model, execute a tool) and report progress.

### What moves

From `storage/run.py` and `storage/policies.py`:

| Python                                                                      | Rust                                                                                                                                                           |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LaneHandleImpl.run`, `_acquire`, `_open_concurrent`, `_commit`, `_abandon` | `run::Lane::open(RunOptions) -> Run`; `Run::commit`, `Run::abandon`; the renewal timer                                                                         |
| `Run`: history, state, agent stack, events, tasks, lease, fence             | `run::Run`: the loaded rows, the write set, the fence; history items and state cross as JSON                                                                   |
| `Tasks`: `write`, `submit`, `wait`, `pending_ask`, `by_message_id`          | `run::Tasks`                                                                                                                                                   |
| `TurnRunner.run_turn` and the policies                                      | `turn::Turn::submit(message) -> task id`, which writes the submitted row, and `turn::Turn::run(task) -> TurnResult`: one state machine, the policy a parameter |
| `_submit`, `_pickup`, `_fail_unrun`                                         | the same names on `turn`                                                                                                                                       |
| `_turn`, `_finish`                                                          | the turn step; the host runs the handler inside it                                                                                                             |
| `Message`, `channel_of`, `resolve_policy`, `resolve_debounce`               | `turn::Message` and the same functions                                                                                                                         |

The turn follows the lane rules of [lanes.md](lanes.md) and
[storage.md](storage.md): runs execute in arrival order, a waiter on an
`interrupt` or `queue` endpoint fails `lease-timeout` after the fixed 60 s
cap, a turn whose lease holder died
fails `worker-lost` and is never re-run, and a failed save ends the task
failed with `livekit/reason=storage-unavailable`.

### The host boundary

One trait, `TurnHost`, with three calls:

```
handle(ctx: TurnContext) -> HandlerOutput   // the a2a_session handler, or Agent.run()
on_update(task_id, text)                    // a working status frame to the wire
on_token(task_id, delta)                    // streamed answer text
```

`TurnContext` carries the thread id, the lane, the message and the batch,
the history as JSON items, the state as JSON, the task id, the lane's task
rows, a delegating caller's history when the message carried one, and the
`Boundary` the handler polls before each model call and before the answer
streams. `HandlerOutput` is what the turn reads back: the output (text or a
typed value), the new items, the state, the usage, an `InputRequired`
question with its pending call, whether the handler yielded, and the
handler's error. Every value crosses as JSON
through the generated records.

The A2A door acks between the two calls: `submit` makes the task durable,
the door sends the `submitted` `Task`, then `run` takes it through its
policy. There is no submitted callback.

In Python, `_core.Turn(thread, lane, host, options)` takes any object with
the three coroutine methods; `turn.submit(message)` returns the task id and
`turn.run(task)` runs it.

### What stays in Python

`protocol.py` (the types a developer reads), `agentdb.py` (the binding over
`_core`), `live.py`, `voice/agent_run.py`, `resolve.py` (thread resolution
is request handling), and the A2A door, which calls `_core.Turn.submit` and
`_core.Turn.run` instead of `TurnRunner`. `rows.py` is generated.

### Order

| Step | Content                                                                                                                                                                           | State                                                                   |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 1    | The lease TTL and the renewal interval as `StorageOptions` fields (default 10 s and 3 s); the Python tests on a real agent-db                                                     | done                                                                    |
| 2    | `run::Lane`, `Run`, `Tasks` in Rust; `run.py` reduced to the binding                                                                                                              | the crate side is built; `run.py` still runs its own unit of work       |
| 3    | `turn` in Rust with `TurnHost`, exposed as `_core.Thread`, `_core.Turn`, `_core.Boundary`; `policies.py` reduced to `Message` and the context type; the A2A door on `_core.Turn` | the crate side and `_core` are built; the door still calls `TurnRunner` |
| 4    | The storage cases as crate tests (`run_local.rs`, `turn_local.rs`)                                                                                                                | done                                                                    |

Each step leaves the demos running.

## The storage binding through `#[host]`

`lk-agents-py/src/storage.rs` and `turn.rs` are hand-written pyo3 over
`lk-agents-storage`. They move to `#[host(..)]` declarations on the storage
crate's own items, expanded in place like every other core crate, so
`lk-agents-py` is back to `lib.rs` and `runtime.rs` and Node gets the same
surface from the same declarations. Requests, results and rows cross as
generated records, not dicts.

## The layout check in CI

The layout check enforces the line cap (1,250 lines per `.rs` file under
`rust/crates`, generated output skipped) and the rule that host records live
only in `lk-agents-abi`. It is ported to this repository and runs in CI, beside
`lk-agents-codegen --check`, so a stale generated `.pyi`, pydantic module or
`.d.ts` fails the build.

## Type enforcement

In CI, `docs/examples` and `examples/` are type-checked with mypy strict
(`tsc` for Node), and `assert_type` / `expectType` tests on the generics
(`Agent[Deps, State, Output]`, `RunContext[Deps, State]`,
`RunResult[Output]`, `AgentTask[Output, TaskState]`).
