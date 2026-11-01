# Python framework ownership inventory

This inventory covers every handwritten `.py` file under `livekit-agents-next/livekit/agents`. Provider plugin packages are intentionally Python and are outside this inventory. Generated implementations, extension stubs and tests are excluded. Line counts are a snapshot, not a measure of remaining framework logic.

Snapshot: 114 files, 28,360 lines. No group below is considered migrated merely because a native implementation already exists; Python callers and duplicate state must also disappear.

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
| [telemetry/__init__.py](../../livekit-agents-next/livekit/agents/telemetry/__init__.py) | 33 |
| [utils/__init__.py](../../livekit-agents-next/livekit/agents/utils/__init__.py) | 23 |
| [version.py](../../livekit-agents-next/livekit/agents/version.py) | 15 |
| [voice/__init__.py](../../livekit-agents-next/livekit/agents/voice/__init__.py) | 52 |

### Boundary generation

Generate record conversion, callback dispatch, views and forwarding from ABI declarations. Keep only Python SDK/object conversion and context capture handwritten.

| File | Lines |
| --- | ---: |
| [_core_view.py](../../livekit-agents-next/livekit/agents/_core_view.py) | 453 |
| [_exceptions.py](../../livekit-agents-next/livekit/agents/_exceptions.py) | 214 |
| [_proto.py](../../livekit-agents-next/livekit/agents/_proto.py) | 208 |
| [a2a/_extension.py](../../livekit-agents-next/livekit/agents/a2a/_extension.py) | 97 |
| [llm/_provider_format.py](../../livekit-agents-next/livekit/agents/llm/_provider_format.py) | 157 |
| [llm/_realtime_options.py](../../livekit-agents-next/livekit/agents/llm/_realtime_options.py) | 45 |
| [llm/chat_context.py](../../livekit-agents-next/livekit/agents/llm/chat_context.py) | 461 |
| [storage/_protocol.py](../../livekit-agents-next/livekit/agents/storage/_protocol.py) | 29 |
| [storage/errors.py](../../livekit-agents-next/livekit/agents/storage/errors.py) | 56 |
| [telemetry/trace_types.py](../../livekit-agents-next/livekit/agents/telemetry/trace_types.py) | 91 |
| [types.py](../../livekit-agents-next/livekit/agents/types.py) | 168 |
| [voice/_session_types.py](../../livekit-agents-next/livekit/agents/voice/_session_types.py) | 103 |
| [voice/turn.py](../../livekit-agents-next/livekit/agents/voice/turn.py) | 98 |

### Provider execution

Audit Python task/retry/fallback ownership against inference/net runtime; move framework pumps and scheduling to Rust. Keep plugin extension methods and actual Python provider calls behind typed ports.

| File | Lines |
| --- | ---: |
| [_bindings.py](../../livekit-agents-next/livekit/agents/_bindings.py) | 390 |
| [inference/eot.py](../../livekit-agents-next/livekit/agents/inference/eot.py) | 194 |
| [inference/llm.py](../../livekit-agents-next/livekit/agents/inference/llm.py) | 440 |
| [inference/realtime.py](../../livekit-agents-next/livekit/agents/inference/realtime.py) | 99 |
| [inference/stt.py](../../livekit-agents-next/livekit/agents/inference/stt.py) | 443 |
| [inference/tts.py](../../livekit-agents-next/livekit/agents/inference/tts.py) | 329 |
| [inference/vad.py](../../livekit-agents-next/livekit/agents/inference/vad.py) | 139 |
| [llm/_realtime.py](../../livekit-agents-next/livekit/agents/llm/_realtime.py) | 285 |
| [llm/duplex.py](../../livekit-agents-next/livekit/agents/llm/duplex.py) | 229 |
| [llm/duplex_adapter.py](../../livekit-agents-next/livekit/agents/llm/duplex_adapter.py) | 100 |
| [llm/fallback_adapter.py](../../livekit-agents-next/livekit/agents/llm/fallback_adapter.py) | 166 |
| [llm/realtime.py](../../livekit-agents-next/livekit/agents/llm/realtime.py) | 107 |
| [llm/realtime_fallback_adapter.py](../../livekit-agents-next/livekit/agents/llm/realtime_fallback_adapter.py) | 121 |
| [llm/remote_chat_context.py](../../livekit-agents-next/livekit/agents/llm/remote_chat_context.py) | 103 |
| [stt.py](../../livekit-agents-next/livekit/agents/stt.py) | 292 |
| [tts.py](../../livekit-agents-next/livekit/agents/tts.py) | 480 |

### Native inference services

Keep Python SDK object hydration and public authoring options; generate options and forward lifecycle/configuration to inference. Check for duplicate provider selection, credentials and lifecycle decisions.

| File | Lines |
| --- | ---: |
| [inference/_utils.py](../../livekit-agents-next/livekit/agents/inference/_utils.py) | 122 |
| [inference/avatar.py](../../livekit-agents-next/livekit/agents/inference/avatar.py) | 210 |
| [inference/interruption.py](../../livekit-agents-next/livekit/agents/inference/interruption.py) | 85 |

### Agent configuration and authoring

Move config validation/defaults, freeze state, tool registration policy and shared definitions to core. Keep callable inspection, decorators, module-name discovery, Pydantic validation and contextvars. Generate routine facade/records.

| File | Lines |
| --- | ---: |
| [llm/utils.py](../../livekit-agents-next/livekit/agents/llm/utils.py) | 481 |
| [voice/agent.py](../../livekit-agents-next/livekit/agents/voice/agent.py) | 731 |
| [voice/agent_host.py](../../livekit-agents-next/livekit/agents/voice/agent_host.py) | 522 |
| [voice/agent_tool.py](../../livekit-agents-next/livekit/agents/voice/agent_tool.py) | 73 |

### Shared toolset lifecycle

Move acquisition identity, reference counts, concurrent setup, rollback, cancellation and last-user close to shared core/runtime. Both text and voice must abort initialization on setup failure. Python retains authored setup/aclose callbacks and schema inspection.

| File | Lines |
| --- | ---: |
| [a2a/remote.py](../../livekit-agents-next/livekit/agents/a2a/remote.py) | 191 |
| [llm/mcp.py](../../livekit-agents-next/livekit/agents/llm/mcp.py) | 291 |
| [llm/tool_context.py](../../livekit-agents-next/livekit/agents/llm/tool_context.py) | 738 |

### Run and text lifecycle

Move resource acquisition, native context/thread setup, stream ownership, cancellation and settlement to shared core/runtime. Keep typed result hydration and invocation-local Python context installation.

| File | Lines |
| --- | ---: |
| [text_session.py](../../livekit-agents-next/livekit/agents/text_session.py) | 227 |
| [voice/agent_run.py](../../livekit-agents-next/livekit/agents/voice/agent_run.py) | 337 |
| [voice/run_stream.py](../../livekit-agents-next/livekit/agents/voice/run_stream.py) | 182 |

### Voice session construction and lifecycle

Remove broad prepare/abort/closed orchestration callbacks and Python-owned resource sequencing. Runtime composes typed agent/provider/room/storage capabilities; voice owns decisions. Generate public facade, hook/node dispatch and event conversion.

| File | Lines |
| --- | ---: |
| [voice/_session_bindings.py](../../livekit-agents-next/livekit/agents/voice/_session_bindings.py) | 525 |
| [voice/_session_callbacks.py](../../livekit-agents-next/livekit/agents/voice/_session_callbacks.py) | 330 |
| [voice/_session_events.py](../../livekit-agents-next/livekit/agents/voice/_session_events.py) | 343 |
| [voice/_session_host.py](../../livekit-agents-next/livekit/agents/voice/_session_host.py) | 172 |
| [voice/agent_session.py](../../livekit-agents-next/livekit/agents/voice/agent_session.py) | 540 |
| [voice/voice_task.py](../../livekit-agents-next/livekit/agents/voice/voice_task.py) | 50 |

### Voice results and events

Use native authoritative state and generated projections for run/event/playback lifecycle. Keep Python assertion/evaluation ergonomics, type hydration and user callback invocation.

| File | Lines |
| --- | ---: |
| [voice/events.py](../../livekit-agents-next/livekit/agents/voice/events.py) | 474 |
| [voice/io.py](../../livekit-agents-next/livekit/agents/voice/io.py) | 423 |
| [voice/run_result.py](../../livekit-agents-next/livekit/agents/voice/run_result.py) | 609 |
| [voice/speech_handle.py](../../livekit-agents-next/livekit/agents/voice/speech_handle.py) | 162 |

### RTC SDK boundary

Keep direct SDK calls and frame/object translation. Generalize reusable RTC source/publication capabilities; move option defaults, playback accounting and all orchestration into voice/runtime. Generate declarations/views.

| File | Lines |
| --- | ---: |
| [utils/participant.py](../../livekit-agents-next/livekit/agents/utils/participant.py) | 136 |
| [voice/room_io.py](../../livekit-agents-next/livekit/agents/voice/room_io.py) | 807 |

### Avatar media and transport

Delete avatar-specific media host protocol. Reuse RTC capabilities for sources/publication; native pacing, bounded queues, interruption accounting and cleanup. Keep user generator/receiver callbacks and SDK frame conversion behind typed ports.

| File | Lines |
| --- | ---: |
| [voice/avatar.py](../../livekit-agents-next/livekit/agents/voice/avatar.py) | 510 |

### Voice auxiliary lifecycle

Move remaining task ownership, speech-state coordination and lifecycle to voice/runtime; retain user callbacks and SDK adapters. Generate configuration and forwarding.

| File | Lines |
| --- | ---: |
| [voice/amd.py](../../livekit-agents-next/livekit/agents/voice/amd.py) | 265 |
| [voice/background_audio.py](../../livekit-agents-next/livekit/agents/voice/background_audio.py) | 285 |
| [voice/remote_session.py](../../livekit-agents-next/livekit/agents/voice/remote_session.py) | 120 |

### Workflows and tool discovery

Move workflow state, ordering, prompt plans and search/tool-discovery policy to core/voice. Retain authored tool bodies, domain type validation and actual Python SDK calls. Generate domain records and facades.

| File | Lines |
| --- | ---: |
| [beta/__init__.py](../../livekit-agents-next/livekit/agents/beta/__init__.py) | 465 |
| [beta/workflows.py](../../livekit-agents-next/livekit/agents/beta/workflows.py) | 538 |
| [voice/tool_executor.py](../../livekit-agents-next/livekit/agents/voice/tool_executor.py) | 121 |

### Storage hydration and execution

Keep Pydantic context serialization, Python dataclass reflection and application-supplied services. Move pending-write/transaction lifetime, thread resolution, turn streams and cleanup sequencing to storage/runtime; generate forwarding and records.

| File | Lines |
| --- | ---: |
| [storage/backends.py](../../livekit-agents-next/livekit/agents/storage/backends.py) | 165 |
| [storage/context.py](../../livekit-agents-next/livekit/agents/storage/context.py) | 350 |
| [storage/context_store.py](../../livekit-agents-next/livekit/agents/storage/context_store.py) | 305 |
| [storage/resolve.py](../../livekit-agents-next/livekit/agents/storage/resolve.py) | 192 |

### A2A and HTTP adapters

Keep ASGI request/response adaptation and authored route/handler registration. Core/net/runtime own authentication decisions, routing state, exchange lifecycle and cancellation; generate configuration and native projections.

| File | Lines |
| --- | ---: |
| [a2a/auth.py](../../livekit-agents-next/livekit/agents/a2a/auth.py) | 150 |
| [a2a/client.py](../../livekit-agents-next/livekit/agents/a2a/client.py) | 5 |
| [a2a/endpoint.py](../../livekit-agents-next/livekit/agents/a2a/endpoint.py) | 292 |
| [a2a/methods.py](../../livekit-agents-next/livekit/agents/a2a/methods.py) | 94 |
| [http.py](../../livekit-agents-next/livekit/agents/http.py) | 304 |

### Process and worker lifecycle

Worker/runtime own registration, leases, job state, process supervision, shutdown/report ordering and timeouts. Keep Python multiprocessing bootstrap, interpreter setup, import/preload operations, SDK room creation and entrypoint callbacks behind narrow ports.

| File | Lines |
| --- | ---: |
| [cli.py](../../livekit-agents-next/livekit/agents/cli.py) | 406 |
| [ipc/__init__.py](../../livekit-agents-next/livekit/agents/ipc/__init__.py) | 450 |
| [ipc/_preload.py](../../livekit-agents-next/livekit/agents/ipc/_preload.py) | 102 |
| [ipc/_preload_forkserver.py](../../livekit-agents-next/livekit/agents/ipc/_preload_forkserver.py) | 15 |
| [ipc/_preload_freeze.py](../../livekit-agents-next/livekit/agents/ipc/_preload_freeze.py) | 40 |
| [ipc/child.py](../../livekit-agents-next/livekit/agents/ipc/child.py) | 558 |
| [ipc/launch.py](../../livekit-agents-next/livekit/agents/ipc/launch.py) | 141 |
| [job.py](../../livekit-agents-next/livekit/agents/job.py) | 1,141 |
| [jupyter.py](../../livekit-agents-next/livekit/agents/jupyter.py) | 117 |
| [worker.py](../../livekit-agents-next/livekit/agents/worker.py) | 500 |

### Telemetry host integration

Retain host OpenTelemetry/Prometheus SDK installation, exporter calls and SDK object translation. Move remaining admission/redaction/buffering/lifecycle policy to core and owned I/O to net/runtime. Never add a Rust telemetry exporter.

| File | Lines |
| --- | ---: |
| [log.py](../../livekit-agents-next/livekit/agents/log.py) | 108 |
| [metrics.py](../../livekit-agents-next/livekit/agents/metrics.py) | 45 |
| [telemetry/_context.py](../../livekit-agents-next/livekit/agents/telemetry/_context.py) | 216 |
| [telemetry/_destination.py](../../livekit-agents-next/livekit/agents/telemetry/_destination.py) | 345 |
| [telemetry/_export.py](../../livekit-agents-next/livekit/agents/telemetry/_export.py) | 324 |
| [telemetry/_providers.py](../../livekit-agents-next/livekit/agents/telemetry/_providers.py) | 701 |
| [telemetry/_redaction.py](../../livekit-agents-next/livekit/agents/telemetry/_redaction.py) | 220 |
| [telemetry/_transport.py](../../livekit-agents-next/livekit/agents/telemetry/_transport.py) | 40 |
| [telemetry/metrics.py](../../livekit-agents-next/livekit/agents/telemetry/metrics.py) | 152 |
| [telemetry/traces.py](../../livekit-agents-next/livekit/agents/telemetry/traces.py) | 448 |
| [telemetry/utils.py](../../livekit-agents-next/livekit/agents/telemetry/utils.py) | 106 |

### Evaluation and simulation

Inference/core own judge execution, aggregation policy and finalize state. Keep application verdicts, authored evaluator callbacks and Python type hydration; generate projections/forwarding.

| File | Lines |
| --- | ---: |
| [evals.py](../../livekit-agents-next/livekit/agents/evals.py) | 470 |
| [simulation.py](../../livekit-agents-next/livekit/agents/simulation.py) | 117 |

### Media and tokenization adapters

Keep public Python iterator/callback interfaces. Media/core own buffering, framing, decode, tokenization and resource state; runtime owns pumps. Keep optional Python library calls (for example blingfire) as provider capabilities.

| File | Lines |
| --- | ---: |
| [tokenize/__init__.py](../../livekit-agents-next/livekit/agents/tokenize/__init__.py) | 299 |
| [tokenize/blingfire.py](../../livekit-agents-next/livekit/agents/tokenize/blingfire.py) | 83 |
| [utils/audio.py](../../livekit-agents-next/livekit/agents/utils/audio.py) | 200 |
| [utils/codecs.py](../../livekit-agents-next/livekit/agents/utils/codecs.py) | 83 |
| [utils/images.py](../../livekit-agents-next/livekit/agents/utils/images.py) | 65 |

### Public Python utilities

Retain generic Python-facing utilities required by provider plugins, including asyncio channel APIs and aiohttp integration. Remove their use for framework runtime ownership; migrate framework transport/retry/pool policy to net/inference rather than porting public asyncio objects into core.

| File | Lines |
| --- | ---: |
| [utils/aio/__init__.py](../../livekit-agents-next/livekit/agents/utils/aio/__init__.py) | 315 |
| [utils/bounded_dict.py](../../livekit-agents-next/livekit/agents/utils/bounded_dict.py) | 90 |
| [utils/connection_pool.py](../../livekit-agents-next/livekit/agents/utils/connection_pool.py) | 201 |
| [utils/http_context.py](../../livekit-agents-next/livekit/agents/utils/http_context.py) | 350 |
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

The inventory is **not an assertion that the whole Python framework has been migrated**. Remaining work includes the session bootstrap protocol (`SessionHost.prepare/started/closed`), default node execution, event/history mirrors, agent callback operation bookkeeping, the remaining avatar session lifecycle, tool-context flattening/discovery, process/job integration and any framework policy still mixed with telemetry/provider SDK integration. These must be migrated at their existing component boundaries; moving their Python bodies to another helper is not completion.

## Validation snapshot

The inventory covers all 114 handwritten framework Python files exactly once. The extension was compiled with `make build`. The stream/lifecycle Python regression set passed 60 tests. The full hermetic gate reported 2,321 passed, 66 skipped and one failure in `test_compressed_finalization_delivers_the_complete_tail_before_join_returns`; the same compressed-audio PCM comparison also fails in isolation. Its exact comparison remains unchanged. Core/voice passed 567 tests and runtime 134; the six native stream regressions passed again after the final EOF fix. Codegen passed 136 tests and the architecture checker fixtures 46. The production architecture gate passed with zero warnings in its 15 warning-clean crates, no new exemptions and no generated drift. Strict mypy still reports 40 existing diagnostics; the comparison reported no new diagnostics from this migration.
