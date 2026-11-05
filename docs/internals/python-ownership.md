# Python framework ownership inventory

This file complements the [Rust core architecture](rust-core.md) with file-level ownership.
A listed file does not imply that every method or cached Python object belongs in Rust.

For the proposed physical layout, see the [Python file cleanup inventory](python-file-layout-inventory.md).
Its target groups handwritten implementations while preserving root application exports.

This inventory covers every handwritten `.py` file under `livekit-agents-next/livekit/agents`. Provider plugin packages are intentionally Python and are outside this inventory. Generated implementations, extension stubs and tests are excluded. Line counts are a snapshot, not a measure of remaining framework logic.

Snapshot (2026-11-01): 114 files, 27,609 lines. Native delegation and Python object views are intentional. Completion requires one policy/lifetime owner, not removal of every Python caller.

## Ownership contract

Rust owns framework policy and resource lifetimes. Synchronous decisions belong in core/voice/media; runtime executes effects through typed capabilities. Existing async I/O crates own their transports, providers, storage and supervision. Python retains language reflection, Pydantic hydration, invocation-local contextvars, user code and actual host SDK calls. ABI declarations generate routine signatures, records, proxies, views and forwarding; arbitrary Python programs embedded in metadata are not a generation strategy.

Toolset setup failure aborts initialization in both text and voice. Acquired resources settle before failure escapes. RTC media uses one shared capability boundary and the existing room connection. No alternate SDK connection or avatar-specific host manager.

## File coverage

### Public exports and Python packaging

Keep Python imports, lazy module loading, entrypoint and package metadata. Generate re-exports where ABI declarations define them.

| File | Lines |
| --- | ---: |
| [__init__.py](../../livekit-agents-next/livekit/agents/__init__.py) | 289 |
| [__main__.py](../../livekit-agents-next/livekit/agents/__main__.py) | 46 |
| [a2a/__init__.py](../../livekit-agents-next/livekit/agents/a2a/__init__.py) | 42 |
| [inference/__init__.py](../../livekit-agents-next/livekit/agents/inference/__init__.py) | 65 |
| [llm/__init__.py](../../livekit-agents-next/livekit/agents/llm/__init__.py) | 152 |
| [resources/__init__.py](../../livekit-agents-next/livekit/agents/resources/__init__.py) | 1 |
| [storage/__init__.py](../../livekit-agents-next/livekit/agents/storage/__init__.py) | 44 |
| [utils/__init__.py](../../livekit-agents-next/livekit/agents/utils/__init__.py) | 23 |
| [version.py](../../livekit-agents-next/livekit/agents/version.py) | 15 |
| [voice/__init__.py](../../livekit-agents-next/livekit/agents/voice/__init__.py) | 52 |

### Boundary generation

Generate record conversion, callback dispatch, views and forwarding from ABI declarations. Keep only Python SDK/object conversion and context capture handwritten.

| File | Lines |
| --- | ---: |
| [_core_view.py](../../livekit-agents-next/livekit/agents/_core_view.py) | 439 |
| [_exceptions.py](../../livekit-agents-next/livekit/agents/_exceptions.py) | 97 |
| [_proto.py](../../livekit-agents-next/livekit/agents/_proto.py) | 208 |
| [a2a/_extension.py](../../livekit-agents-next/livekit/agents/a2a/_extension.py) | 97 |
| [llm/_provider_format.py](../../livekit-agents-next/livekit/agents/llm/_provider_format.py) | 142 |
| [llm/_realtime_options.py](../../livekit-agents-next/livekit/agents/llm/_realtime_options.py) | 45 |
| [llm/chat_context.py](../../livekit-agents-next/livekit/agents/llm/chat_context.py) | 461 |
| [storage/_protocol.py](../../livekit-agents-next/livekit/agents/storage/_protocol.py) | 29 |
| [storage/errors.py](../../livekit-agents-next/livekit/agents/storage/errors.py) | 23 |
| [types.py](../../livekit-agents-next/livekit/agents/types.py) | 34 |
| [voice/_session_types.py](../../livekit-agents-next/livekit/agents/voice/_session_types.py) | 82 |
| [voice/turn.py](../../livekit-agents-next/livekit/agents/voice/turn.py) | 86 |

### Provider execution

Native inference owns attempts, retry/fallback selection and stream admission. Python executes authored provider methods and SDK feed/drain tasks. Shared aio helpers retain unfinished cleanup; native close proceeds independently of stalled host input cleanup. Generated provider APIs remain supported.

| File | Lines |
| --- | ---: |
| [_bindings.py](../../livekit-agents-next/livekit/agents/_bindings.py) | 641 |
| [inference/eot.py](../../livekit-agents-next/livekit/agents/inference/eot.py) | 194 |
| [inference/llm.py](../../livekit-agents-next/livekit/agents/inference/llm.py) | 402 |
| [inference/realtime.py](../../livekit-agents-next/livekit/agents/inference/realtime.py) | 99 |
| [inference/stt.py](../../livekit-agents-next/livekit/agents/inference/stt.py) | 330 |
| [inference/tts.py](../../livekit-agents-next/livekit/agents/inference/tts.py) | 267 |
| [inference/vad.py](../../livekit-agents-next/livekit/agents/inference/vad.py) | 139 |
| [llm/_realtime.py](../../livekit-agents-next/livekit/agents/llm/_realtime.py) | 285 |
| [llm/duplex.py](../../livekit-agents-next/livekit/agents/llm/duplex.py) | 198 |
| [llm/duplex_adapter.py](../../livekit-agents-next/livekit/agents/llm/duplex_adapter.py) | 87 |
| [llm/fallback_adapter.py](../../livekit-agents-next/livekit/agents/llm/fallback_adapter.py) | 164 |
| [llm/realtime.py](../../livekit-agents-next/livekit/agents/llm/realtime.py) | 38 |
| [llm/realtime_fallback_adapter.py](../../livekit-agents-next/livekit/agents/llm/realtime_fallback_adapter.py) | 117 |
| [llm/remote_chat_context.py](../../livekit-agents-next/livekit/agents/llm/remote_chat_context.py) | 33 |
| [stt.py](../../livekit-agents-next/livekit/agents/stt.py) | 297 |
| [tts.py](../../livekit-agents-next/livekit/agents/tts.py) | 480 |

### Native inference services

Keep Python SDK object hydration and public authoring options; generate options and forward lifecycle/configuration to inference. Check for duplicate provider selection, credentials and lifecycle decisions.

| File | Lines |
| --- | ---: |
| [inference/_utils.py](../../livekit-agents-next/livekit/agents/inference/_utils.py) | 122 |
| [inference/avatar.py](../../livekit-agents-next/livekit/agents/inference/avatar.py) | 186 |
| [inference/interruption.py](../../livekit-agents-next/livekit/agents/inference/interruption.py) | 38 |

### Agent configuration and authoring

Core owns configuration validation, freeze checks, hook replacement and retry resolution. Python keeps callable inspection, decorators, module-name discovery, Pydantic validation, InstructionsView and contextvars. Generated records/facades expose the native configuration.

| File | Lines |
| --- | ---: |
| [llm/utils.py](../../livekit-agents-next/livekit/agents/llm/utils.py) | 472 |
| [voice/agent.py](../../livekit-agents-next/livekit/agents/voice/agent.py) | 731 |
| [voice/agent_host.py](../../livekit-agents-next/livekit/agents/voice/agent_host.py) | 522 |
| [voice/agent_tool.py](../../livekit-agents-next/livekit/agents/voice/agent_tool.py) | 73 |

### Shared toolset lifecycle

Core/runtime own acquisition identity, reference counts, concurrent setup, rollback, cancellation and last-user close. Search/proxy toolsets declare hidden resource dependencies to that group. Python retains authored setup/aclose callbacks, tool identity and schema inspection.

| File | Lines |
| --- | ---: |
| [a2a/remote.py](../../livekit-agents-next/livekit/agents/a2a/remote.py) | 191 |
| [llm/mcp.py](../../livekit-agents-next/livekit/agents/llm/mcp.py) | 339 |
| [llm/tool_context.py](../../livekit-agents-next/livekit/agents/llm/tool_context.py) | 616 |

### Run and text lifecycle

Runtime RunResources owns thread/context/toolset acquisition, rollback and release. Native RunStream owns stream admission, cancellation and settlement. Python retains typed result/exception hydration and invocation-local context installation.

| File | Lines |
| --- | ---: |
| [text_session.py](../../livekit-agents-next/livekit/agents/text_session.py) | 227 |
| [voice/agent_run.py](../../livekit-agents-next/livekit/agents/voice/agent_run.py) | 337 |
| [voice/run_stream.py](../../livekit-agents-next/livekit/agents/voice/run_stream.py) | 182 |

### Voice session construction and lifecycle

Voice owns decisions; runtime SessionOwner owns startup sequencing, acquisition, rollback and close. SessionHost constructs Python SDK/models/context views through explicit callbacks. Event/hook dispatch and Python object identity remain adapters, not a second session owner.

| File | Lines |
| --- | ---: |
| [voice/_session_bindings.py](../../livekit-agents-next/livekit/agents/voice/_session_bindings.py) | 528 |
| [voice/_session_callbacks.py](../../livekit-agents-next/livekit/agents/voice/_session_callbacks.py) | 328 |
| [voice/_session_events.py](../../livekit-agents-next/livekit/agents/voice/_session_events.py) | 343 |
| [voice/_session_host.py](../../livekit-agents-next/livekit/agents/voice/_session_host.py) | 170 |
| [voice/agent_session.py](../../livekit-agents-next/livekit/agents/voice/agent_session.py) | 540 |
| [voice/voice_task.py](../../livekit-agents-next/livekit/agents/voice/voice_task.py) | 44 |

### Voice results and events

Use native authoritative state and generated projections for run/event/playback lifecycle. Keep Python assertion/evaluation ergonomics, type hydration and user callback invocation.

| File | Lines |
| --- | ---: |
| [voice/events.py](../../livekit-agents-next/livekit/agents/voice/events.py) | 344 |
| [voice/io.py](../../livekit-agents-next/livekit/agents/voice/io.py) | 405 |
| [voice/run_result.py](../../livekit-agents-next/livekit/agents/voice/run_result.py) | 596 |
| [voice/speech_handle.py](../../livekit-agents-next/livekit/agents/voice/speech_handle.py) | 159 |

### RTC SDK boundary

Runtime owns room orchestration and playback accounting through the shared RTC capability boundary. Python retains direct SDK calls, frame/object translation, publication options and callback registration. Generated boundary extraction reads resolved RoomOptions directly; no intermediate field dictionary is maintained.

| File | Lines |
| --- | ---: |
| [utils/participant.py](../../livekit-agents-next/livekit/agents/utils/participant.py) | 136 |
| [voice/room_io.py](../../livekit-agents-next/livekit/agents/voice/room_io.py) | 630 |

### Avatar media and transport

Runtime reuses RTC source/publication capabilities and owns pacing, bounded frame delivery, interruption accounting and cleanup. Python retains SDK frame conversion and user generator/receiver callbacks; the retired avatar-specific media host protocol is not needed.

| File | Lines |
| --- | ---: |
| [voice/avatar.py](../../livekit-agents-next/livekit/agents/voice/avatar.py) | 507 |

### Voice auxiliary lifecycle

Native voice/runtime own AMD admission, playback selection and remote-session decisions. Python retains completion/result views, user callbacks and SDK sources/sinks. Rejected AMD attachment cannot replace or close the active detector.

| File | Lines |
| --- | ---: |
| [voice/amd.py](../../livekit-agents-next/livekit/agents/voice/amd.py) | 268 |
| [voice/background_audio.py](../../livekit-agents-next/livekit/agents/voice/background_audio.py) | 285 |
| [voice/remote_session.py](../../livekit-agents-next/livekit/agents/voice/remote_session.py) | 120 |

### Workflows and tool discovery

Core/voice own workflow ordering, prompt plans and tool-search ranking. Python retains authored tool bodies, domain validation, reflection and SDK calls. Toolset dependencies use the shared native group even when the visible search/proxy tool list hides a child.

| File | Lines |
| --- | ---: |
| [beta/__init__.py](../../livekit-agents-next/livekit/agents/beta/__init__.py) | 437 |
| [beta/workflows.py](../../livekit-agents-next/livekit/agents/beta/workflows.py) | 507 |
| [voice/tool_executor.py](../../livekit-agents-next/livekit/agents/voice/tool_executor.py) | 64 |

### Storage hydration and execution

Native storage/runtime own transactions, revisions, thread resolution and run-resource lifetimes. Python retains Pydantic/dataclass hydration and application services. The checkpoint bridge invokes the authored handler through a shared channel, with terminal errors and retained close settlement.

| File | Lines |
| --- | ---: |
| [storage/backends.py](../../livekit-agents-next/livekit/agents/storage/backends.py) | 165 |
| [storage/context.py](../../livekit-agents-next/livekit/agents/storage/context.py) | 363 |
| [storage/context_store.py](../../livekit-agents-next/livekit/agents/storage/context_store.py) | 305 |
| [storage/resolve.py](../../livekit-agents-next/livekit/agents/storage/resolve.py) | 192 |

### A2A and HTTP adapters

Keep ASGI request/response adaptation and authored route/handler registration. Core/net/runtime own authentication decisions, routing state, exchange lifecycle and cancellation; generate configuration and native projections.

| File | Lines |
| --- | ---: |
| [a2a/auth.py](../../livekit-agents-next/livekit/agents/a2a/auth.py) | 75 |
| [a2a/client.py](../../livekit-agents-next/livekit/agents/a2a/client.py) | 5 |
| [a2a/endpoint.py](../../livekit-agents-next/livekit/agents/a2a/endpoint.py) | 292 |
| [a2a/methods.py](../../livekit-agents-next/livekit/agents/a2a/methods.py) | 94 |
| [http.py](../../livekit-agents-next/livekit/agents/http.py) | 282 |

### Process and worker lifecycle

Worker/runtime own registration, leases, job state, process supervision, shutdown/report ordering and timeouts. Keep Python multiprocessing bootstrap, interpreter setup, import/preload operations, SDK room creation and entrypoint callbacks behind narrow ports.

| File | Lines |
| --- | ---: |
| [cli.py](../../livekit-agents-next/livekit/agents/cli.py) | 406 |
| [ipc/__init__.py](../../livekit-agents-next/livekit/agents/ipc/__init__.py) | 445 |
| [ipc/_preload.py](../../livekit-agents-next/livekit/agents/ipc/_preload.py) | 102 |
| [ipc/_preload_forkserver.py](../../livekit-agents-next/livekit/agents/ipc/_preload_forkserver.py) | 15 |
| [ipc/_preload_freeze.py](../../livekit-agents-next/livekit/agents/ipc/_preload_freeze.py) | 40 |
| [ipc/child.py](../../livekit-agents-next/livekit/agents/ipc/child.py) | 558 |
| [ipc/launch.py](../../livekit-agents-next/livekit/agents/ipc/launch.py) | 141 |
| [ipc/prometheus.py](../../livekit-agents-next/livekit/agents/ipc/prometheus.py) | 70 |
| [job.py](../../livekit-agents-next/livekit/agents/job.py) | 1,110 |
| [worker.py](../../livekit-agents-next/livekit/agents/worker.py) | 498 |

### Telemetry host integration

Rust owns spans, metrics, report records, export and the session report upload ([telemetry internals](telemetry.md)). Python keeps one module: the OpenTelemetry API provider over the native span, the bridge for an application's SDK provider, `configure`, `set_capture_content` and `diagnostics`, the event loop monitor and the framework's spans around Python-only code (`room_connect`, `wait_for_participant`, `rpc_call`, `rpc_handler`). The worker's Prometheus process metrics are `ipc/prometheus.py`.

| File | Lines |
| --- | ---: |
| [log.py](../../livekit-agents-next/livekit/agents/log.py) | 108 |
| [metrics.py](../../livekit-agents-next/livekit/agents/metrics.py) | 45 |
| [telemetry.py](../../livekit-agents-next/livekit/agents/telemetry.py) | 397 |

### Evaluation and simulation

Inference/core own judge execution, aggregation policy and finalize state. Keep application verdicts, authored evaluator callbacks and Python type hydration; generate projections/forwarding.

| File | Lines |
| --- | ---: |
| [evals.py](../../livekit-agents-next/livekit/agents/evals.py) | 394 |
| [simulation.py](../../livekit-agents-next/livekit/agents/simulation.py) | 111 |

### Media and tokenization adapters

Keep public Python iterator/callback interfaces. Media/core own buffering, framing, decode, tokenization and resource state; runtime owns pumps. Keep optional Python library calls (for example blingfire) as provider capabilities.

| File | Lines |
| --- | ---: |
| [tokenize/__init__.py](../../livekit-agents-next/livekit/agents/tokenize/__init__.py) | 296 |
| [tokenize/blingfire.py](../../livekit-agents-next/livekit/agents/tokenize/blingfire.py) | 83 |
| [utils/audio.py](../../livekit-agents-next/livekit/agents/utils/audio.py) | 200 |
| [utils/codecs.py](../../livekit-agents-next/livekit/agents/utils/codecs.py) | 83 |
| [utils/images.py](../../livekit-agents-next/livekit/agents/utils/images.py) | 38 |

### Public Python utilities

Python-facing channels, asyncio task references and aiohttp hooks remain host adapters. Shared cancellation retains tasks until settlement and exposes pending cleanup. Native ConnectionPoolState owns reuse/retirement; Python holds actual SDK connections. Generic user-data containers and dynamic user keys remain valid.

| File | Lines |
| --- | ---: |
| [utils/aio/__init__.py](../../livekit-agents-next/livekit/agents/utils/aio/__init__.py) | 371 |
| [utils/connection_pool.py](../../livekit-agents-next/livekit/agents/utils/connection_pool.py) | 201 |
| [utils/http_context.py](../../livekit-agents-next/livekit/agents/utils/http_context.py) | 358 |
| [utils/misc.py](../../livekit-agents-next/livekit/agents/utils/misc.py) | 131 |

## Completion criteria

- Remove the Python owner and duplicate state when migrating a responsibility; splitting files does not count.
- Shared Agent and toolset contracts serve both text and voice.
- Test failure, cancellation, concurrent reuse, rollback and close at native boundaries, plus public Python compatibility.
- Run generation checks, architectural checks and relevant Python/Rust suites. Keep this inventory honest about unfinished work.

## Migration ledger

Implemented ownership changes (validation is recorded with the change):

- Avatar media uses shared RTC source/publication capabilities. `_AvatarMedia` and its host proxy are removed. Voice owns the pacing plan; runtime owns publication, bounded frame delivery and cleanup. The generated native synchronizer preserves the operations exposed by `runner.av_sync`; its concrete type is native rather than `rtc.AVSynchronizer`.
- Queue audio admission, segment epochs, discarded-segment ordering and stale playback report handling are in `voice::avatar::queue`. Python forwards frames and executes playback event effects; it owns no segment queue.
- Shared toolset lifetime and group/scope selection are in `core::toolsets`; runtime owns setup/close effects, cancellation and rollback. Python refcounts, locks and separate text/voice acquisition/rollback loops are removed. Object identity, rather than the public toolset label, identifies a shared connection.
- Voice startup, handoff rollback, per-agent release and session shutdown address the native toolset registry directly. Python provides SDK toolset identities and schema reflection; `_core_setup_tools`, `_core_close_toolsets` and the bootstrap abort callback are removed. Failed or cancelled admissions do not occupy a later activation’s release slot.
- Owned toolset setup uses generated HostInvoker cancellation signalling and waits for the Python task’s actual settlement, including async `finally` blocks.
- Direct agent runs and whole text-session turns use one native `RunResources` lifetime for temporary thread storage, context activation and toolset leases. Python ownership flags, per-agent acquisition/release loops and `AsyncExitStack` cleanup are removed. Borrowed application resources remain borrowed; native close ends owned activations and releases Python hydration through a narrow SDK callback. Text and voice share core start/close admission.
- Request streams now bind directly to native runner sources. Core owns consumer/read admission, text snapshots, event envelopes and presentation acknowledgments; runtime owns cancellation and source/resource settlement. Public event records and cursor bindings are generated. Python retains result/exception hydration and language protocol forwarding, with no stream scheduler, consumer flags, text buffer or release map.
- Native cleanup failures are cached as the public stream’s Python exception outcome. Repeated result retrieval and context exit do not invoke cleanup again or consume a captured exception twice.
- Agent authoring configuration, callback ID order and hook replacement, freeze checks, retry resolution and configuration projection are in `core::agent_authoring`. `AgentRetries` and `ModelSettings` are generated. Python keeps callable registries, module-name discovery, Pydantic/schema reflection and callback invocation.
- Lifecycle hooks and session views carry the agent scope identity separately from its name. Distinct agents sharing a name retain their own hooks and resources; failed activation does not replace the current agent.

The inventory is **not an assertion that the whole Python framework has been migrated**.
Source-reviewed responsibilities above deliberately retain SDK, reflection and hydration code.
The RoomOptions dictionary projection, AuthGrant/Principal constructor projections and fixed
provider member probes have been removed. The lasting contracts are documented under
[shared binding semantics](rust-core.md#shared-binding-semantics).
Provider review is sampled; no claim is made that every plugin has been manually certified.

## Validation snapshot

The table covers all handwritten framework files exactly once. Behavior changes are checked
through Cargo's freshly linked Python contracts plus native ownership tests. Current execution,
transport, report and MCP changes have scoped strict mypy checks. Architecture/code-generation
checks remain separate from behavioral tests. Historical whole-repository test counts are not
presented as validation of this checkout.
