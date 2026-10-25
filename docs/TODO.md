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

## Rust core

| Doc section | Code change |
| --- | --- |
| [rust-core.md](rust-core.md): packaging | Build `livekit.agents._core` as `cp310-abi3` inside the livekit-agents wheel (it is built per Python version today). |
| [rust-core.md](rust-core.md): records only in abi | `lk-agents-storage/src/schema.rs` declares host records outside `lk-agents-abi`. |
| [rust-core.md](rust-core.md): bindings | Node bindings expand in place instead of `lk-agents-node/src/generated`; hand-written files in `lk-agents-node`; `lk-agents-worker/src/python.rs` is hand-written pyo3. |
| [rust-core.md](rust-core.md): LiveKit crates | Storage hand-signs its JWT and inference mints one with `jsonwebtoken`; both use `livekit-token`. |
| [rust-core.md](rust-core.md): a core is a state machine | `lk-agents-voice` decides inside spawned tasks (generation with its tools, IVR, AMD, fillers, the synchroniser, keyterms, room I/O, the end-of-turn leg: about 40 sites in 19 files); `Driver::run` awaits a `select!` instead of `handle(event, now)` and `wake_at()`; `execute` performs decisions and reads the clock. Order in [planned-rust-core.md](planned-rust-core.md#migration-order). |
| [rust-core.md](rust-core.md): events and effects are data | Providers and the realtime session go by id (`lk_agents_abi::resources`), but `Command` still carries output, input and recorder objects (step 1: an interrupt cuts the sink and reads its playhead in the decision); a reply resolves its LLM and TTS through the table at creation and the keyterm pass calls `update_keyterms` through it, until steps 2 and 3 make them core state; `Answer::Speech` carries a shared `SpeechHandle` until the host view lands; the generation still settles its tools on `oneshot` acks (`GenEvent::ToolsFinished`, the deferred tool start). The scheduling pause, the close's drain, handoff and drain answer by signal; API calls are answered by call id. |
| [rust-core.md](rust-core.md): the host reads a view | `SpeechHandle` is shared state (`watch`, `Mutex`, atomics) the host reads and writes from its own thread; `resolved_options`, `steps`, `linked_participant` read the driver directly. |
| [rust-core.md](rust-core.md): a method earns its place | Left for the steps that rewrite them: `Tools` still bundles running calls, the tool reply and fillers, and `SpeechScheduler` the preemptive slot (generation, step 2); `AgentActivity`'s six realtime fields are not one `Option<RealtimeLeg>`; `spawn_drain` is called with the same eight arguments four times (`io.rs`, `builder.rs`); `StreamKind` duplicates `AudioChannel`; `diagnose` is a free function beside `Out`. |
| [rust-core.md](rust-core.md): a core is deterministic | The voice tasks still read the clock (generation, AMD, IVR, the synchroniser, keyterms); ids come from `rand` in `lk-agents-storage` and `lk-agents-inference`; the layout check does not reject clock reads or `RandomState` yet. |
| [rust-core.md](rust-core.md): tests drive the core | `lk-agents-fuzz` owns the queue and steps `handle`, but the driver's spawned tasks still run on tokio: the policy can only interleave whole task polls with core steps, and needs a one-task event interval to do it. |
| [rust-core.md](rust-core.md): what decides is a core | `lk-agents-storage`'s lease, fence and turn policy and `lk-agents-inference`'s reconnect, retry and fallback decide inside awaited scripts; `WorkerState` uses `tick` instead of `Event::Wake` and `wake_at()`. |
| [rust-core.md](rust-core.md): only `lk-agents-runtime` depends on an async runtime | Every product crate depends on tokio, `lk-agents-media` through `lk-agents-abi`; `lk-agents-runtime` does not exist yet (I/O is spread over the cores, `lk-agents-rtc` and the bindings' `runtime.rs`); the layout check has no `cargo tree` rule. Layout in [planned-rust-core.md](planned-rust-core.md#crates). |
| [rust-core.md](rust-core.md), [planned-rust-core.md](planned-rust-core.md): typed end to end | CI runs no cargo, codegen or type tests; no `assert_type` tests. |
| [planned-rust-core.md](planned-rust-core.md) | `storage.rs` and `turn.rs` in `lk-agents-py` move to `#[host]` on `lk-agents-storage`; the layout check is ported and runs in CI; `Turn` splits into `submit` and `run`; the wait timeout is a fixed 60 s. |
