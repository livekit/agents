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
| [agent-tasks.md](agent-tasks.md): hooks | `on_exit(self, ctx)` with `ctx` a `RunContext`, and called in text when a task pops; `on_enter` takes it today, `on_exit` does not. |

## Threads, lanes and storage

| Doc section | Code change |
| --- | --- |
| [threads.md](threads.md): no storage | Without storage, a run's `ctx.thread` is an in-memory thread with the same API. |
| [journal.md](journal.md) | Write the voice session's own events on a call; every other journal event is written today. |
| [channels.md](channels.md): attachments | File and image parts become typed chat content; they are JSON text today. |

## A2A, delegation and voice

| Doc section | Code change |
| --- | --- |
| [voice.md](voice.md): a call per voice turn | When the lane's text door is not `live`, a text between two voice turns reaches the session's history on reload, not a realtime model's own context. |
| [delegation.md](delegation.md): when the expert asks a question | A question cut off under the `interrupt` or `queue` inbox shows the question to the client of the task that asked (only `live` holds its frames back). |
| [delegation.md](delegation.md): when the expert asks a question | Work of several tasks that ends together is answered by one reply: a task among them that waited on a question and whose work asked nothing ends `completed`, `answeredWith`, and its question is dropped (a single resumed task asks it again after its reply). |
| [delegation.md](delegation.md): where work runs | A holder on another worker learns of an arrival or a cancel by polling the thread's database (every second while work runs in the background, a call every two seconds); the request waits for its task the same way. Routing by the node itself is cloud work. |

## Rust core

| Doc section | Code change |
| --- | --- |
| [rust-core.md](rust-core.md): packaging | Build `livekit.agents._core` as `cp310-abi3` inside the livekit-agents wheel (it is built per Python version today). |
| [rust-core.md](rust-core.md): records only in abi | `lk-agents-storage/src/schema.rs` declares host records outside `lk-agents-abi`. |
| [rust-core.md](rust-core.md): bindings | Node bindings expand in place instead of `lk-agents-node/src/generated`; hand-written files in `lk-agents-node`; `lk-agents-worker/src/python.rs` is hand-written pyo3. |
| [rust-core.md](rust-core.md): LiveKit crates | Storage hand-signs its JWT and inference mints one with `jsonwebtoken`; both use `livekit-token`. |
| [rust-core.md](rust-core.md), [planned-rust-core.md](planned-rust-core.md): typed end to end | CI runs no cargo, codegen or type tests; no `assert_type` tests. |
| [planned-rust-core.md](planned-rust-core.md) | `storage.rs` and `turn.rs` in `lk-agents-py` move to `#[host]` on `lk-agents-storage`; the layout check is ported and runs in CI; `Turn` splits into `submit` and `run`; the wait timeout is a fixed 60 s. |
