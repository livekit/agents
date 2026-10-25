# TODO: where the code does not match these docs yet

The docs in this folder describe the approved design. Each item below is a
place where the code on this branch still differs, with the doc section it
belongs to. An item leaves this file when the code lands.

## Agents and tools

| Doc section | Code change |
| --- | --- |
| [agents.md](agents.md): what an agent is not | Remove `Agent.update_options`; tests in `test_agent_api.py` use it. |
| [agents.md](agents.md): `RunContext` | `ctx.thread` is the durable thread in every run (`RunContext` has no `thread` today); add `ctx.live_session` for the call. |
| [agents.md](agents.md): `RunContext` | `ctx.task_id` is the A2A task id (today a `run_...` id, or a speech handle id on a call). |
| [agents.md](agents.md): structured output | Wrap a non-object `output_type` as `{"result": ...}` in the final-result tool and unwrap it; lists and scalars fail today. |
| [agents.md](agents.md): output validators | A validator's return value replaces the output; it is discarded today. |
| [tools.md](tools.md): `ModelRetry` | Handle `ModelRetry` from a tool on a call like `agent.run()` does; it becomes an internal error today. |
| [tools.md](tools.md): toolsets | `AgentRun` calls `setup()` on toolsets each turn and never `aclose()`; an `MCPToolset` built per run is never closed. |

## Agent tasks and application context

[agent-tasks.md](agent-tasks.md) defines conversational tasks using shared Agent
configuration. [state-and-deps.md](state-and-deps.md) defines context persistence.
Durable execution and [code workflows](code-workflows.md) are deferred.

- Migrate to Agent[Context], RunContext[Context], run(context=...), and
  ctx.context, retaining output_type instead of separate deps_type/state_type.
- Implement typed persisted fields, explicit store.load_context/save_context,
  and runtime-owned ctx.store lifecycle. Save only marked application data;
  reconstruct runtime services explicitly.
- Specify context scopes, field validation, schema changes, revisions, and
  automatic managed saves with history. Do not promise crash-safe external effects.
- Support intermediate conversational text alongside typed final output,
  automatic InputRequired, output functions, and explicit string finalization.
- Implement Call(agent, context=...), shared versus isolated child context,
  and parent tool-result delivery. Migrate existing AgentTask examples.
- Keep direct application records and context fields on the same AgentDB
  where appropriate; define transaction ownership before exposing writes.
- Verify typing, context loading/saving, human input, approval invalidation,
  output validation, and child completion.

Schedulers, external-operation waits, automatic recovery, operation journals,
and deployment routing are not implementation requirements for this phase.
The earlier exploration is archived at `~/livekit/durable-execution.md`.

## Threads, lanes and storage

| Doc section | Code change |
| --- | --- |
| [threads.md](threads.md): no storage | Without storage, a run's `ctx.thread` is an in-memory thread with the same API. |
| [journal.md](journal.md) | Write the voice session's own events on a call; every other journal event is written today. |
| [channels.md](channels.md): attachments | File and image parts become typed chat content; they are JSON text today. |

## A2A, delegation and voice

| Doc section | Code change |
| --- | --- |

## Rust core

| Doc section | Code change |
| --- | --- |
| [rust-core.md](rust-core.md): packaging | Build `livekit.agents._core` as `cp310-abi3` inside the livekit-agents wheel (it is built per Python version today). |
| [rust-core.md](rust-core.md): records only in abi | `lk-agents-storage/src/schema.rs` declares host records outside `lk-agents-abi`. |
| [rust-core.md](rust-core.md): bindings | Node bindings expand in place instead of `lk-agents-node/src/generated`; hand-written files in `lk-agents-node`; `lk-agents-worker/src/python.rs` is hand-written pyo3. |
| [rust-core.md](rust-core.md): LiveKit crates | Storage hand-signs its JWT and inference mints one with `jsonwebtoken`; both use `livekit-token`. |
| [rust-core.md](rust-core.md), [planned-rust-core.md](planned-rust-core.md): typed end to end | CI runs no cargo, codegen or type tests; no `assert_type` tests. |
| [planned-rust-core.md](planned-rust-core.md) | `storage.rs` and `turn.rs` in `lk-agents-py` move to `#[host]` on `lk-agents-storage`; the layout check is ported and runs in CI; `Turn` splits into `submit` and `run`; the wait timeout is a fixed 60 s. |
