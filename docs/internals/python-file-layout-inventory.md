# Handwritten Python file cleanup inventory

Reduce the framework from **113 to 63 handwritten Python files**: **50 fewer files, a 44%
reduction**. Keep the existing names and root application imports; consolidate implementations
by feature, move incidental internal imports directly, and retire forwarding leaves.
Generated output and stubs are outside this inventory and contribute no savings.

The final proposal retains useful public namespaces: metrics, types, llm.realtime and
llm.duplex remain real modules. llm.remote_chat_context leaves the public API. Root and feature imports both remain supported.
Utility cleanup is an explicit exception: the four-file utils layout intentionally replaces
old utility leaf paths. File counts follow that compatibility choice, rather than defining it.

This target includes entire package consolidations, shared execution modules and small
facade removals. It preserves real provider namespaces, optional integrations and process
entrypoints where they serve a concrete purpose. The proposal changes physical layout and
some documented import paths; it does not change AgentRun policy, voice scheduling, tool
execution, cancellation or resource ownership.

Scope: every handwritten `.py` under `livekit-agents-next/livekit/agents`, including the new
telemetry diagnostics file. The November 1, 2026 snapshot contains **27,412 source lines**.
Plugins, tests, examples and scripts are consumers to migrate, rather than packages to merge.
This is an implementation plan; runtime import safety must be verified during the moves.
The [Python ownership inventory](python-ownership.md) describes layer ownership separately.
The [deprecated and legacy API inventory](python-legacy-api-inventory.md) adds the concrete
API removal work: 43 removal groups, five retirement decisions and seven active API groups
to retain. Argument/alias removals do not count as additional framework file savings.

## Public API and file layout

Application code should continue using the primary public API:

```python
from livekit.agents import Agent, ChatContext, RunContext, VoiceSession, function_tool
```

Root imports are a convenient default, not the only supported form. Application and provider
code can also use real public namespaces such as `from livekit.agents import llm, stt, tts`
and `from livekit.agents.llm import ChatContext, RealtimeModel`. Public namespace support
does not require preserving every internal implementation file.

The earlier 2.0 exports (`6b1b4a5ea`) and current examples already follow that pattern.
An implementation living under llm or voice does not require applications to import it there.
Moving that implementation does not require changing its class name or root export.

Use the existing names for common tool-authoring options at the root too: propose exporting
Toolset, AsyncToolset, AsyncToolOptions, ToolHandlingOptions and UpdatePromptArgs from
`livekit.agents`. Their definitions/records stay with the tool implementation; the package
root simply exports the same objects. Application docs should use these root imports.
This is an export proposal, not a new set of wrapper classes.

Provider authors retain real `llm`, `stt`, `tts` and inference modality namespaces. Those
namespaces contain overlapping names such as FallbackAdapter and provider option types.
They need not retain the current number of files underneath them. The target llm initializer
contains the generic LLM contract and its fallback adapter directly. The realtime and duplex
namespaces each contain their generic contract and adapter implementation.

Do not preserve an import merely because it appears in an internal caller or test. Migrate
those consumers. For documented imports, the table below specifies the replacement and the
feature-document update needed before implementation. No compatibility files, renamed-type
aliases, fake namespaces, sys.modules registrations or custom import loaders are part of
the target. Ordinary public re-exports and existing lazy loading of real optional modules
remain useful.

## Exact file reduction

| Component | Current files | Target files | Net reduction |
| --- | ---: | ---: | ---: |
| Other root files, excluding metrics, stt and tts | 16 | 13 | 3 |
| A2A | 7 | 1 | 6 |
| Prebuilt beta tools and workflows | 2 | 1 | 1 |
| Storage | 7 | 3 | 4 |
| Telemetry and the retained root metrics file | 13 | 7 | 6 |
| Utilities | 9 | 4 | 5 |
| Resource package marker | 1 | 0 | 1 |
| LLM contracts and adapters | 14 | 8 | 6 |
| Voice and shared agent execution adapters | 24 | 12 | 12 |
| Hosted inference | 10 | 6 | 4 |
| Process execution and preload | 6 | 4 | 2 |
| Tokenizers | 2 | 2 | 0 |
| STT and TTS interfaces/adapters | 2 | 2 | 0 |
| **Total** | **113** | **63** | **50** |

Count unique final physical files, not the number of source rows marked for a merge.
The mapping retires **55 existing file paths** and introduces **5 replacement paths**, for
**50 net fewer files**. Source moves with two destinations do not create duplicate savings.
Moving implementation lines is not a claim that those lines disappear.

## Complete target layout

Only handwritten Python files are shown. The resources directory retains its assets without
an __init__.py. Generated output, stubs and plugin packages remain outside this tree.

```text
livekit/agents/
  __init__.py
  __main__.py
  _bindings.py
  _proto.py
  a2a.py
  beta.py
  cli.py
  evals.py
  http.py
  inference/__init__.py
  inference/avatar.py
  inference/llm.py
  inference/realtime.py
  inference/stt.py
  inference/tts.py
  ipc/__init__.py
  ipc/_preload.py
  ipc/_preload_forkserver.py
  ipc/child.py
  job.py
  llm/__init__.py
  llm/_realtime.py
  llm/chat_context.py
  llm/duplex.py
  llm/mcp.py
  llm/realtime.py
  llm/tool_context.py
  llm/utils.py
  log.py
  metrics.py
  storage/__init__.py
  storage/context.py
  storage/session.py
  stt.py
  telemetry/__init__.py
  telemetry/_destination.py
  telemetry/_export.py
  telemetry/_providers.py
  telemetry/metrics.py
  telemetry/traces.py
  text_session.py
  tokenize/__init__.py
  tokenize/blingfire.py
  tts.py
  types.py
  utils/__init__.py
  utils/aio.py
  utils/http_context.py
  utils/media.py
  version.py
  voice/__init__.py
  voice/_session_bindings.py
  voice/_session_callbacks.py
  voice/agent_run.py
  voice/agent_session.py
  voice/amd.py
  voice/avatar.py
  voice/background_audio.py
  voice/events.py
  voice/io.py
  voice/room_io.py
  voice/run_result.py
  worker.py
```

## LLM from fourteen files to eight

| Final file | Current files grouped here | Approximate input lines |
| --- | --- | ---: |
| `llm/__init__.py` | Root and generic LLM fallback | 316 |
| `llm/chat_context.py` | Chat context | 435 |
| `llm/duplex.py` | Public duplex contract and its realtime adapter | 285 |
| `llm/realtime.py` | Retained generic realtime contract and its fallback adapter | 155 |
| `llm/tool_context.py` | Tools/toolsets and public tool-handling option/prompt imports | 623 |
| `llm/utils.py` | Schema/reflection/serialization helpers and provider-format conversion | 614 |
| `llm/_realtime.py` | Concrete OpenAI-compatible realtime and its option helper | 330 |
| `llm/mcp.py` | MCP integration | 339 |

The root implements the generic LLM contract and fallback, and exports the realtime and
duplex modules’ generic contracts. Define the duplex adapter after its bases in duplex.py. Define realtime fallback after its bases in realtime.py.
Keep `llm.RealtimeModel` as the generic provider base. The concrete OpenAI-compatible class
with the same short name remains in `_realtime`; the move must not select a provider when
someone imports the generic base. Keep the existing `RealtimeModelFallbackAdapter` spelling.

Define/import the generic bases before subclasses. Load chat and tools before adapter
definitions, then serialization helpers after those types exist. Chat/tool conversion
methods call the utils conversion helpers at operation time: blindly replacing their eager
_provider_format imports with eager utils imports would create a cycle because utils imports
chat and tool types. The provider-conversion helpers use direct same-module calls after
the merge. MCP and the concrete realtime constructor stay outside eager generic imports.

Update annotation globals too: ChatContext currently refers to provider-format record types,
and ToolContext refers to TOOL_BEHAVIOR through the removed module. Import authoritative
record types directly and keep the existing pure type alias with the tool declarations.
Runtime type-hint resolution must work without restoring an eager conversion-utils import.

Remove `llm.remote_chat_context` and its native class: it was a server-history mirror for the
Azure Voice Live plugin, which now runs on the shared Rust realtime path as the `voice_live`
dialect (the beta protocol behind an Azure endpoint). Nothing else uses it.
Native-backed records/events whose declared public module names a retired leaf must point
to the retained public home. Update authoritative declarations and regenerate normally;
generated files themselves are not consolidation targets.

## Voice from twenty four files to twelve

| Final file | Current files grouped here | Approximate input lines |
| --- | --- | ---: |
| `voice/__init__.py` | Initializer, Agent authoring and local-agent tool descriptor | 857 |
| `voice/agent_run.py` | Run driver, shared AgentCallbacks/argument/output validation, template renderer | 925 |
| `voice/agent_session.py` | Session and VoiceTask | 583 |
| `voice/_session_bindings.py` | Bindings/configuration, startup host, signature aliases/sampler, remote console binding | 914 |
| `voice/_session_callbacks.py` | Application hook/node adapters and native event/telemetry dispatch | 677 |
| `voice/events.py` | Public events and RunContext | 364 |
| `voice/io.py` | Media interfaces, streaming turn-detector protocols and SpeechHandle | 650 |
| `voice/run_result.py` | Result variants/assertions/capture and RunStream lifecycle | 789 |
| `voice/room_io.py` | Concrete livekit.rtc adapter | 630 |
| `voice/avatar.py` | Avatar integration | 499 |
| `voice/amd.py` | Answering-machine detection | 268 |
| `voice/background_audio.py` | Background playback/assets | 282 |

Keep clear sections for the responsibilities within each merged file. The native AgentRun
remains the shared decision owner for text and voice; putting its Python driver and shared
callbacks together removes file hops without introducing another execution loop.

The voice initializer can implement Agent authoring directly. Initialize low-level IO/room
interfaces, then event/RunContext types, then authoring prerequisites and Agent plus its
private tool descriptor, and finally session/features. IO and room adapters keep Agent and
VoiceSession imports type-only or local. Agent execution methods retain their local imports
of agent_run. `text_session.py` updates its existing top-level Agent/descriptor import;
argument/schema helpers retain their existing local imports.

Move SpeechHandle to IO rather than events: generated event definitions import the handle,
so placing the handle in events would create a declaration cycle. Update that authoritative
host-type import before removing speech_handle.py. Turn options remain root exports; their
protocols live beside the other host ports in IO.

Define session aliases before binding consumers and keep VoiceSession type-only there.
ConsoleCall remains a distinct class with its one process-level claim; moving it beside
session startup does not create a second console owner. Hook and native-event dispatch
remain distinct sections in one callback file, with handlers defined before dispatch tables.
RunContext.update keeps its renderer import local because agent_run imports RunContext.

The merged result module preserves one-consumer streaming, cancellation/settlement and
captured voice result behavior. VoiceTask remains session-owned with the same temporary
floor/restoration semantics. Keep concrete room, avatar, AMD and background playback files:
these are substantial integrations, rather than forwarding layers.

## Services and shared helpers

| Final owner | Current files grouped here | Approximate input lines |
| --- | --- | ---: |
| `a2a.py` | All seven A2A files | 794 |
| `beta.py` | Prebuilt tools and workflows | 944 |
| `storage/__init__.py` | Configuration/backends, errors, Thread/Lane handles, history projection | About 430 |
| `storage/context.py` | Persisted application context, formerly context_store.py | 302 |
| `storage/session.py` | Request-scoped A2ASessionContext/native Turn host, formerly context.py | 363 |
| `telemetry/__init__.py` | Cheap context, carriers, constants, PII helpers, diagnostics and job attribution | About 580 |
| `telemetry/_export.py` | SDK admission/bounding, immutable snapshots, redaction and exporter wrappers | 511 |
| `telemetry/metrics.py` | OTel/Prometheus metric collection | 153 |
| `metrics.py` | Retained public provider metric records and log_metrics | 45 |
| `telemetry/_providers.py` | Process providers and Cloud pipelines, minus cheap attribution helper | About 675 |
| `telemetry/_destination.py` | Per-job destination/auth/gating and early-log retention | 386 |
| `telemetry/traces.py` | Session report and RTC RPC integration | 455 |
| `utils/__init__.py` | Small shared utilities and participant/track waits | 253 |
| `utils/aio.py` | Async channels/tasks/cancellation and connection pool | 569 |
| `utils/media.py` | Audio frames/files, audio decoder and image/frame-reader adaptation | 215 |
| `utils/http_context.py` | Provider HTTP/SSL/WebSocket lifetime | 367 |
| `_bindings.py` | Model method bodies, host proxy views and exception hooks | 1,178 |
| `_proto.py` | Shared wire conversion and storage item payload conversion | About 230 |
| `evals.py` | Judges and simulation context/verdict authoring | 505 |
| `__init__.py` | Root exports, including existing shared types | 289 plus added exports |
| `types.py` | Retained public constants, connection options and NotGivenOr | 34 |

Input sizes sum current handwritten content before import/docstring deduplication and
redistribution of shared imports. They are planning budgets, not enforced Python file limits.

### A2A and prebuilt tools become real modules

A2A's current initializer already eagerly loads its auth, endpoint and remote pieces.
One 794-line feature module can contain credentials/grants, the endpoint's native service,
card/metadata adaptation and delegation tools in named sections. Preserve native auth,
endpoint close, question correlation and cancellation semantics. Different classes keep
their distinct responsibilities; they do not need seven files to do so.

Likewise beta.py contains the existing prebuilt tool families and capture/DTMF/transfer
workflows in sections. It replaces the beta package and its workflows import directly;
do not retain a workflows namespace object. Preserve VoiceTask floor ownership, transfer
cleanup and background audio shutdown.

### Storage has three explicit responsibilities

The storage root owns configuration, error exports and durable thread/lane handles.
Export errors and define configuration/handle prerequisites before dependent context imports;
keep session host imports out of that initialization path. Backend reconstruction and
borrowed-handle close behavior remain intact.

Use context.py for persisted application context and session.py for request execution.
These are different responsibilities despite the old overlapping context filenames.
Perform the two moves together so imports cannot accidentally resolve to the wrong class.
Move item_to_payload/payload_to_item to the shared _proto leaf; move history_of to the
storage root. Keep LLM construction/imports local where the protocol/storage leaf needs them.

Storage errors currently advertise storage.errors and native conversion imports it.
Canonicalize the declaration to storage root and update exception conversion, metadata and
supported pickle tests. Backend _rebuild/__reduce__ references also move to the root.
Those dependencies require migration work, not a retained handwritten facade.

### Telemetry has one cheap root and five concrete integrations

Keep the root free of SDK exporter/provider imports. Define constants, diagnostics counters,
export-suppression context and cheap helpers before importing anything that consumes them.
Importing a context or reading diagnostics must not initialize OTLP exporters. Replace
trace_types/utils/_context namespace hops with direct names, rather than fake module objects.

Move the pure invocation-local _job_stamp_attributes helper from providers to this cheap
root. Early log retention currently reaches the heavy provider module for that lookup;
the new direct call should preserve attribution without forcing process pipeline setup.
Cheap record_exception calls the already-present set_error_type helper directly.

Merge SDK export admission and redaction: these share the SDK record boundary and already
call one another. Preserve copy-before-redaction, native destination policy, limits and
bounded cleanup. Keep destination/early-buffer behavior separate from process provider
construction, and keep report/RPC and metric collection as their actual integrations.

Keep provider metric records/log_metrics in the existing public metrics.py module.
Keep telemetry.metrics for actual OTel/Prometheus collection. These responsibilities remain
distinct, and existing `from livekit.agents import metrics` and `from livekit.agents.metrics
import STTMetrics` keep their module/class identity. Register collection instruments once
and avoid importing exporter/provider construction when merely loading public records.

### Utilities follow their actual feature

The proposed utility cleanup is **9 files → 4**, with intentional migration of the old
misc/participant/connection_pool/audio/codecs/images leaf paths. Keep the familiar aio
namespace and existing helper/class names; do not require callers to import utility
primitives from the agents root.

```text
utils/
  __init__.py       # Common helpers, participant waits, public exports
  aio.py            # Channels, intervals, tasks, cleanup and ConnectionPool
  media.py          # Audio frames/files, decoder and image encoding
  http_context.py   # HTTP sessions, SSL and WebSocket lifetime
```

`utils/aio/__init__.py` is already the package's only source file. Moving it to aio.py
removes an unnecessary directory while retaining the normal `livekit.agents.utils.aio`
import path. That move alone does not reduce the Python file count; absorbing
connection_pool.py removes one file. Keep `utils.ConnectionPool` as an ordinary export of
that same class, alongside `utils.aio.ConnectionPool`. Existing `utils.aio.Chan`, Interval,
TaskSet, cancel_and_wait and cleanup helper behavior remains intact. Migrate the redundant
gracefully_cancel spelling to cancel_and_wait before removing that alias, as recorded in F04
of the legacy API inventory; preserve its cancellation/settlement semantics.

Retain ConnectionPool functionality: direct constructors occur in 20 plugin source files
across 17 provider packages. It pools Python-owned WebSockets and SDK clients, including
Google speech clients and Vakyam streaming sessions. Rust ConnectionPoolState already owns
reuse, expiry and invalidation bookkeeping; the Python adapter holds the actual objects and
executes their connect/close callbacks. Native transport pooling for hosted inference does
not replace those plugin SDK resources. Consolidate the implementation file into aio.py,
preserving utils.ConnectionPool as the plugin-facing API.

After merging the pool, remove its `from . import aio` self-dependency and call the cleanup
helper directly. Update direct connection_pool imports and test monkeypatch targets to the
real aio module. Preserve pool callback, timeout, invalidation and close behavior, and the
retained cleanup semantics of cancelled waiters. The current aio initializer has only
standard-library imports; pool imports retain their current depth relative to utils.

Root utilities define foundational functions before child imports; participant-to-room
resolution stays local. aio.py replaces a one-file package and absorbs ConnectionPool,
preserving task/cancellation cleanup, TTL, weakrefs and pool close behavior. media.py combines
small SDK media helpers with clear audio and image sections. Utils already eagerly loads
these helpers; retain exactly-once frame-reader registration and decoder/file-task lifetime.
HTTP context remains its own request/session/SSL/WebSocket owner, separate from ASGI serving.

### Root adapters and asset packaging

_bindings becomes the one handwritten native-host adaptation file: existing method bodies,
proxy views and exception-format hooks. It is imported while generated bases are being
declared, so remove eager LLM/voice/io/tool lookups from the moved code. Use type-only
annotations and operation-local class imports. The existing cached _views factory can do
its IO/Room lookup at call time. No new registry or global initialization protocol is needed.
Update declared host view/proxy paths and exception hook/public identities before deletion.

Keep the public types.py module and its existing root re-exports. Foundational type and
binding initialization must still precede defaults that invoke validation. Preserving this
small public module avoids unnecessary constant/annotation import migrations; use the same
authoritative classes, rather than replacements or module registries.

SimulationContext joins evals, retaining its root export. Keep LLM annotations type-only,
concrete judge/tool imports inside their operations, telemetry use inside evaluation, and
JobContext lookup type-only/local. Loading simulation context must not start judge/provider
pipelines. Job construction retains its existing local import boundary.

Remove resources/__init__.py while retaining all audio assets. Use the already-existing
parent package as the resource anchor:

```python
files("livekit.agents").joinpath("resources", self.value)
```

Preserve as_file/ExitStack lifetime. Maturin already explicitly includes
`livekit/agents/resources/*`; verify built-wheel contents and installed-wheel clip access.
A second Python package marker is unnecessary for that lookup.

## Hosted inference and process execution

| Final owner | Current files grouped here | Approximate input lines |
| --- | --- | ---: |
| `inference/__init__.py` | Root, shared credentials/headers, VAD, end-of-turn and interruption constructors | 558 |
| `inference/llm.py` | LLM implementation/options | 406 |
| `inference/stt.py` | STT implementation/options | 330 |
| `inference/tts.py` | TTS implementation/options | 267 |
| `inference/avatar.py` | Avatar integration, lazily loaded | 186 |
| `inference/realtime.py` | Realtime integration, lazily loaded | 99 |
| `ipc/__init__.py` | Parent executor and launch configuration/partial-process ownership | 580 |
| `ipc/child.py` | Child execution and handshake | 558 |
| `ipc/_preload.py` | Shared once-per-image warmup | 95 |
| `ipc/_preload_forkserver.py` | Ordered forkserver preload and final freeze | 59 |

Define shared gateway credentials/headers and detector prerequisites before inference leaf
imports. Local EOT/Silero imports stay inside their model factories. Keep avatar and realtime
independently lazy: their voice dependencies must not be loaded while voice initializes.
LLM/STT/TTS remain real option namespaces because several provider types have the same names.
Retire detector leaf imports and update their declared event/error identities and labels.

In IPC, define shared launch helpers in the parent before importing child code. Child and
forkserver code use those initialized helpers directly. Preserve partial-Popen registration
and handle ownership even on launch failure. Child remains an importable process entry with
its native proxy/pickle identity.

The forkserver bootstrap performs all framework/application preloads, then gc.collect,
then Python/native thread inspection, then gc.freeze. Refuse unsafe or uninspectable images.
The separate freeze phase file disappears; the once-per-image warmup entry remains.
Migrate both proc_bench.py's preload list and test_preload_launch.py's runpy source-path test.
Use configured forkserver launch, rather than substituting an unconfigured bootstrap string.

## Concrete import replacements

These are the target import contracts, including proposed updates to documented deep paths.
Feature documents change with the implementation, after the export choices are settled.
Applications keep using root names; provider and integration code uses the real retained
namespace. Update authoritative class/proxy/error paths and regenerate where required.

| Existing use | Target use |
| --- | --- |
| `from livekit.agents import Agent, ChatContext, RunContext, VoiceTask` | Unchanged |
| `from livekit.agents.llm import Toolset, AsyncToolset` in application code | `from livekit.agents import Toolset, AsyncToolset`; llm exports remain for provider consumers |
| `from livekit.agents.voice.tool_executor import AsyncToolOptions` | `from livekit.agents import AsyncToolOptions`; implementation/export records belong in llm.tool_context |
| `from livekit.agents.voice.agent import Agent` | `from livekit.agents import Agent`; internal voice code imports from its package |
| `from livekit.agents.llm.realtime import RealtimeModel` | Unchanged; llm-root exports remain supported too |
| `from livekit.agents.llm.duplex_adapter import DuplexRealtimeAdapter` | `from livekit.agents.llm.duplex import DuplexRealtimeAdapter`; llm-root export remains |
| `llm.remote_chat_context.RemoteChatContext(...)` | Removed; no remaining consumer |
| `from livekit.agents.llm._provider_format import convert_mid_conversation_instructions` | Import the same helper from llm.utils |
| `from livekit.agents.a2a.auth import Credentials` | `from livekit.agents.a2a import Credentials` |
| `from livekit.agents.a2a.endpoint import a2a_session` | `from livekit.agents import a2a_session`; integration code can use agents.a2a |
| `from livekit.agents.storage.errors import StorageError` | `from livekit.agents.storage import StorageError` |
| `from livekit.agents.storage.backends import SQLiteStorage` | `from livekit.agents.storage import SQLiteStorage` |
| `from livekit.agents.storage.context_store import load_context` | `from livekit.agents import load_context`; implementation is storage.context |
| `from livekit.agents.storage.context import A2ASessionContext` | `from livekit.agents.a2a import A2ASessionContext`; implementation is storage.session |
| `from livekit.agents import SimulationContext` | Unchanged; implementation moves to evals |
| `from livekit.agents.types import NOT_GIVEN, NotGivenOr` | Unchanged; existing root exports remain supported |
| `from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT` | Unchanged |
| Other types-module exports: ATTRIBUTE_PUBLISH_ON_BEHALF, USERDATA_TTS_STARTED_TIME, TranscriptChunk, FlushSentinel, connection options | Existing public module and root exports remain supported |
| `from livekit.agents.metrics import STTMetrics` | Unchanged |
| `telemetry.trace_types.ATTR_*` or `telemetry.utils.record_exception` | Direct imports of the same constants/helpers from telemetry |
| `utils.codecs.AudioStreamDecoder`, `utils.audio.audio_frames_from_file`, `utils.images.encode` | Same names under the real utils.media module |
| `utils.ConnectionPool` | Unchanged root export; implementation joins utils.aio |
| `from livekit.agents.inference.vad import VAD` | `from livekit.agents.inference import VAD` |
| `from livekit.agents.inference.interruption import AdaptiveInterruptionDetector` | `from livekit.agents.inference import AdaptiveInterruptionDetector` |
| `from livekit.agents.beta.workflows import <type>` | Import the same type directly from livekit.agents.beta |
| `from livekit.agents.voice.run_stream import RunStream` | `from livekit.agents import RunStream`; implementation joins voice.run_result |

The [models](../models.md), [tools](../tools.md), [usage](../usage.md),
[agents](../agents.md), [server](../server.md), [clients](../clients.md),
[testing](../testing.md) and [voice session](../voice-session.md) documents need the corresponding
import examples/type tables updated when these paths move. Also migrate plugin attribute
access, native host-type references, annotation strings, monkeypatch targets, process import
strings and scripts; an AST import search alone will miss some of these consumers.

## Files retained for concrete reasons

The 63-file target includes deliberately substantial modules. Keeping a module here is not
a promise to retain every current function or import in it.

- Keep actual process entries and phase boundaries: __main__, child, warmup, forkserver.
- Keep version.py as release-managed metadata; this cleanup does not change versions.
- Keep metrics.py, types.py, llm.realtime and llm.duplex as useful public
  modules, preserving familiar v1 imports rather than replacing them with compatibility shims.
- Keep stt.py and tts.py: actual provider adapters with colliding same-named helpers.
- Keep the generic versus concrete realtime distinction and the independent MCP integration.
- Keep IO interfaces separate from their concrete room adapter; retain avatar, AMD and
  background playback as recognizable integrations.
- Keep telemetry context cheap and process SDK construction lazy. Destination buffering,
  SDK record export and report/RPC projection retain explicit lifecycle responsibilities.
- Keep tokenize's two files for now: the abstract and concrete SentenceTokenizer share a
  short name. Removing the remaining leaf requires a constructor/naming contract change,
  rather than an incidental import migration.
- Keep the existing complete job, worker, CLI, ASGI transport, logging, wire-codec and text
  session owners. Combining them would make startup/resource boundaries harder to inspect.

## Additional reductions beyond sixty three files

These options remove one more file each, but are outside the selected target. They need a
separate justification beyond their numerical saving; they are not hidden requirements for
the 63-file plan.

| Option | Further reduction | Concrete cost |
| --- | ---: | --- |
| Tokenizers 2 → 1 | 1 | Make the existing root SentenceTokenizer implement the current concrete default while preserving custom subclasses, or select a concrete name. The first changes abstractness/constructor behavior; the second introduces a name. Do not silently replace the base contract. |
| IPC 4 → 3 | 1 | Combine parent and child in an approximately 1,138-line owner; migrate child native proxy and process identities. It combines two process-side lifetimes, even though physical grouping is possible. |
| LLM 8 → 7 | 1 | Combine tools and conversion utils, around 1,237 input lines. It mixes chat/image serialization into tool authoring; preserve the retained public realtime and remote-chat namespaces. |
| Telemetry 6 → 5 | 1 | Combine providers and destination, about 1,083 input lines before moving attribution. Localize heavy construction dependencies so early log retention does not load process OTLP providers sooner. Preserve per-job gating and process cleanup as distinct classes. |
| Inference 6 → 5 | 1 | Put avatar/realtime integrations in one lazy sessions file. It couples two independently requested dependencies solely to remove one file. |

Whole-package llm.py, voice.py, utils.py or telemetry.py remain technically possible with
public-path changes, but combine several thousand lines or unrelated owners. The selected
layout removes fragmentation aggressively without hiding it behind giant files or runtime
namespaces.

## Implementation sequence and checks

### Deprecated and legacy APIs to remove

Apply the [full removal inventory](python-legacy-api-inventory.md) with the physical moves.
It records each old symbol, exact source, replacement, callers and checks; useful v1 public
namespaces and supported SDK/data contracts are explicitly retained.

- [ ] F01–F04: remove redundant job/MCP/buffer aliases and migrate the duplicate cancellation
  spelling as part of utils cleanup.
- [ ] F05–F08: remove ignored native-backed HTTP-session, VAD executor and buffered-token
  formatting arguments. Preserve working Python-provider injection and tokenizer options.
- [ ] F09–F11: remove below-floor RTC feature fallbacks and obsolete codecs/images extras,
  migrating plugin manifests and checking installed-wheel behavior.
- [ ] P01–P04: remove hosted-realtime and Sarvam aliases, fixed-execution Silero knobs and
  the unused v1 Spatius warmup argument.
- [ ] P05–P10: remove deprecated Deepgram/ElevenLabs aliases/options and obsolete Cartesia
  record/update/encoding surfaces, migrating every constructor and update entrypoint.
- [ ] P11–P18: remove discarded Speechify/Gnani kwargs and old Phonic/SLNG configuration
  spellings, without removing current per-tool options or actual endpoint state.
- [ ] P19–P27: remove AssemblyAI and Speechmatics deprecated inputs/modes/no-op methods,
  preserving supported diarization, forwarding and endpoint behavior.
- [ ] P28–P32: remove deprecated Krisp constructor arguments, Tavus identifiers/env/persona
  endpoint and retired direct-Anthropic model suggestions after their caller migrations.
- [ ] D02–D05: retire old RTC/provider protocols, stored-usage decoding and Krisp direct
  method shims only after their explicit server/client/data/lifecycle prerequisites.

### File regrouping and verification

1. Preserve public types/metrics imports and establish cheap utility/telemetry definitions
   before their consumers.
   Consolidate _bindings with runtime lookups deferred; update authoritative hook/proxy paths.
2. Consolidate llm contracts, chat/tools/conversion and inference helpers with the dependency
   order above. Check generic and concrete provider subclassing in fresh interpreters.
3. Group voice authoring, run callbacks, results, IO and session collaborators. Preserve
   both voice and text use of native AgentRun, callback binding and event ordering.
4. Move storage context/session paths together, flatten A2A and beta, and migrate direct
   consumers. Check native errors, supported pickle reconstruction and borrowed handles.
5. Merge utilities and SDK export code; retain the public metric record module, remove the
   resource marker and consolidate IPC helpers/phases. Preserve privacy, early logs, cancellation and cleanup.
6. Update feature docs, root exports, plugin imports, scripts and authoritative metadata for
   every retired path. Regenerate normally without hand-editing output. Delete retired
   handwritten files after consumers move; create no compatibility shims.

Use existing focused behavior contracts, rather than tests that merely mirror file moves.
Check cold imports in multiple orders, optional extras, provider model construction,
Agent.as_tool/TextSession and VoiceSession startup, shared AgentCallbacks, VoiceTask floor
restoration, RunStream cancellation/settlement, media registration/decoder closure,
telemetry consent/redaction/early buffering, spawn/forkserver partial-launch ownership and
built-wheel clip loading. Run Python lint/type checks and affected native binding contracts
against the Cargo-linked module. Check codegen whenever declared paths move.

## Complete current file to target inventory

Every one of the **113 current handwritten files** appears once below. Destinations are real
physical modules; split moves are explicit. “Retain” means the same physical path continues,
usually with related code absorbed. “Merge” retires the source path into its owner. “Move”
relocates an owner, and “Remove marker” deletes the Python marker while retaining assets.
These source classifications are not additive savings; the exact target contains 63 files.

### Package root

| Current file | Lines | Action | Target and reason |
| --- | ---: | --- | --- |
| [__init__.py](../../livekit-agents-next/livekit/agents/__init__.py) | 289 | Retain | `__init__.py`. Public application exports; retain both root and feature namespace imports, drawing shared types from the retained types module. |
| [__main__.py](../../livekit-agents-next/livekit/agents/__main__.py) | 46 | Retain | `__main__.py`. Required python -m livekit.agents entry file; moving main elsewhere cannot eliminate it. |
| [_bindings.py](../../livekit-agents-next/livekit/agents/_bindings.py) | 642 | Retain | `_bindings.py`. One handwritten native-host adaptation owner: generated model method bodies, host proxy views, and exception formatting hooks. |
| [_core_view.py](../../livekit-agents-next/livekit/agents/_core_view.py) | 439 | Merge | `_bindings.py`. Proxy views are handwritten native-host adaptations just like existing binding helpers; a separate 439-line file is not required. |
| [_exceptions.py](../../livekit-agents-next/livekit/agents/_exceptions.py) | 97 | Merge | `_bindings.py`. Exception method bodies are generated-model hooks and fit the same binding-helper owner; public errors themselves remain authoritative generated classes reexported at root. |
| [_proto.py](../../livekit-agents-next/livekit/agents/_proto.py) | 208 | Retain | `_proto.py`. Retain the shared protobuf/JSON leaf; absorb storage item payload conversion. |
| [cli.py](../../livekit-agents-next/livekit/agents/cli.py) | 406 | Retain | `cli.py`. Keep the Python process/asyncio launcher as one concrete lifecycle owner. |
| [evals.py](../../livekit-agents-next/livekit/agents/evals.py) | 394 | Retain | `evals.py`. One evaluation API contains judges and simulation scenario/user verdict context. |
| [http.py](../../livekit-agents-next/livekit/agents/http.py) | 804 | Retain | `http.py`. Keep complete ASGI HTTP/WebSocket transport; do not merge client HTTP session handling into server protocol state. |
| [job.py](../../livekit-agents-next/livekit/agents/job.py) | 1124 | Retain | `job.py`. Keep job/resource owner. It refers to simulation context from evals without absorbing unrelated judge behavior. |
| [log.py](../../livekit-agents-next/livekit/agents/log.py) | 108 | Retain | `log.py`. Keep import-time Python log levels/logger/filter setup independent of SDK exporter/provider loading. |
| [metrics.py](../../livekit-agents-next/livekit/agents/metrics.py) | 45 | Retain | `metrics.py`. Retain the v1 public provider metric records and log_metrics API; OTel/Prometheus collection stays separate. |
| [simulation.py](../../livekit-agents-next/livekit/agents/simulation.py) | 111 | Merge | `evals.py`. Simulation verdict/context and judge evaluation form one evaluation/testing API; avoid adding this to already-large job.py. |
| [stt.py](../../livekit-agents-next/livekit/agents/stt.py) | 298 | Retain | `stt.py`. Already consolidated public STT contract, fallback and stream adaptation module. |
| [text_session.py](../../livekit-agents-next/livekit/agents/text_session.py) | 227 | Retain | `text_session.py`. Keep request-scoped TextSession host/agent hydration wrapper distinct from A2A transport Turn context. |
| [tts.py](../../livekit-agents-next/livekit/agents/tts.py) | 480 | Retain | `tts.py`. Already consolidated public TTS contract, emitter/stream adaptation and pacing module. |
| [types.py](../../livekit-agents-next/livekit/agents/types.py) | 34 | Retain | `types.py`. Retain useful v1 public constants, sentinel/options types and NotGivenOr, plus existing root re-exports. |
| [version.py](../../livekit-agents-next/livekit/agents/version.py) | 15 | Retain | `version.py`. Keep release metadata file, changed only in release PRs. |
| [worker.py](../../livekit-agents-next/livekit/agents/worker.py) | 541 | Retain | `worker.py`. Keep server/handler registration and worker callbacks separate from job execution and ASGI protocol resources. |

### a2a

| Current file | Lines | Action | Target and reason |
| --- | ---: | --- | --- |
| [a2a/__init__.py](../../livekit-agents-next/livekit/agents/a2a/__init__.py) | 42 | Move | `a2a.py`. Single A2A authoring/service module replaces the package: auth, endpoint lifecycle, delegation tools and public generated/native exports. |
| [a2a/_extension.py](../../livekit-agents-next/livekit/agents/a2a/_extension.py) | 95 | Move | `a2a.py`. Endpoint metadata/card adaptation belongs with the endpoint, in the same A2A feature module. |
| [a2a/auth.py](../../livekit-agents-next/livekit/agents/a2a/auth.py) | 75 | Move | `a2a.py`. Credentials/grants and authorization errors are a named section of the single A2A feature module. |
| [a2a/client.py](../../livekit-agents-next/livekit/agents/a2a/client.py) | 5 | Move | `a2a.py`. Delete pure native reexport hop; directly import AgentClient/Reply in the one A2A module. |
| [a2a/endpoint.py](../../livekit-agents-next/livekit/agents/a2a/endpoint.py) | 292 | Move | `a2a.py`. Endpoint configuration, HTTP routes and native A2AService lifetime become one visible section in the A2A module. |
| [a2a/methods.py](../../livekit-agents-next/livekit/agents/a2a/methods.py) | 94 | Move | `a2a.py`. Service exists only to implement A2AEndpoint; keep their lifecycle together. |
| [a2a/remote.py](../../livekit-agents-next/livekit/agents/a2a/remote.py) | 191 | Move | `a2a.py`. RemoteAgent tools are the client-facing part of the same A2A authoring API. |

### beta

| Current file | Lines | Action | Target and reason |
| --- | ---: | --- | --- |
| [beta/__init__.py](../../livekit-agents-next/livekit/agents/beta/__init__.py) | 437 | Move | `beta.py`. One prebuilt-tools module for end-call, tool search/proxy, DTMF and capture/transfer workflows. |
| [beta/workflows.py](../../livekit-agents-next/livekit/agents/beta/workflows.py) | 507 | Move | `beta.py`. Workflow tools and prebuilt tools have the same author-facing beta component and fit in one file. |

### inference

| Current file | Lines | Action | Target and reason |
| --- | ---: | --- | --- |
| [inference/__init__.py](../../livekit-agents-next/livekit/agents/inference/__init__.py) | 65 | Retain | `inference/__init__.py`. Implement common gateway policy and the native-backed default detectors directly in the canonical inference module, alongside its provider facade and lazy avatar/realtime exports. |
| [inference/_utils.py](../../livekit-agents-next/livekit/agents/inference/_utils.py) | 122 | Merge | `inference/__init__.py`. Move shared gateway credentials, URL/header and quota helpers into the canonical inference module before provider imports, reducing a dedicated helper file without duplicating policy. |
| [inference/avatar.py](../../livekit-agents-next/livekit/agents/inference/avatar.py) | 186 | Retain | `inference/avatar.py`. Keep the real avatar integration as one lazily loaded feature. |
| [inference/eot.py](../../livekit-agents-next/livekit/agents/inference/eot.py) | 194 | Merge | `inference/__init__.py`. Inline the 194-line end-of-turn detector beside the other canonical detector constructors; remove the incidental eot module. |
| [inference/interruption.py](../../livekit-agents-next/livekit/agents/inference/interruption.py) | 38 | Merge | `inference/__init__.py`. Inline the 38-line adaptive interruption constructor beside VAD and TurnDetector; remove its nested facade. |
| [inference/llm.py](../../livekit-agents-next/livekit/agents/inference/llm.py) | 406 | Retain | `inference/llm.py`. Keep the documented LLM model/option namespace and real streaming/configuration implementation. |
| [inference/realtime.py](../../livekit-agents-next/livekit/agents/inference/realtime.py) | 99 | Retain | `inference/realtime.py`. Keep the stateful realtime model integration as a lazily loaded feature. |
| [inference/stt.py](../../livekit-agents-next/livekit/agents/inference/stt.py) | 330 | Retain | `inference/stt.py`. Keep the documented STT provider namespace and STT/SpeechStream implementation. |
| [inference/tts.py](../../livekit-agents-next/livekit/agents/inference/tts.py) | 267 | Retain | `inference/tts.py`. Keep the documented TTS provider namespace and TTS/SynthesizeStream implementation. |
| [inference/vad.py](../../livekit-agents-next/livekit/agents/inference/vad.py) | 139 | Merge | `inference/__init__.py`. Inline the native-backed VAD wrapper and its stream beside the other default detectors. |

### ipc

| Current file | Lines | Action | Target and reason |
| --- | ---: | --- | --- |
| [ipc/__init__.py](../../livekit-agents-next/livekit/agents/ipc/__init__.py) | 439 | Retain | `ipc/__init__.py`. Keep the parent-side executor and absorb its process-launch support for one complete parent lifecycle owner. |
| [ipc/_preload.py](../../livekit-agents-next/livekit/agents/ipc/_preload.py) | 95 | Retain | `ipc/_preload.py`. Keep the intentional shared once-per-image warmup entry, including the child-only active initialization helpers. |
| [ipc/_preload_forkserver.py](../../livekit-agents-next/livekit/agents/ipc/_preload_forkserver.py) | 19 | Retain | `ipc/_preload_forkserver.py`. Keep one fail-closed forkserver bootstrap and absorb its final image-freeze phase. |
| [ipc/_preload_freeze.py](../../livekit-agents-next/livekit/agents/ipc/_preload_freeze.py) | 40 | Merge | `ipc/_preload_forkserver.py`. The 40-line freeze phase belongs at the end of the ordered forkserver bootstrap, not in a separate imported module. |
| [ipc/child.py](../../livekit-agents-next/livekit/agents/ipc/child.py) | 558 | Retain | `ipc/child.py`. Keep child-side handshake, signal policy, host invocation and shutdown together as the other real process boundary. |
| [ipc/launch.py](../../livekit-agents-next/livekit/agents/ipc/launch.py) | 141 | Merge | `ipc/__init__.py`. Absorb process configuration, validation and partial-Popen ownership into the parent executor that owns launched processes. |

### llm

| Current file | Lines | Action | Target and reason |
| --- | ---: | --- | --- |
| [llm/__init__.py](../../livekit-agents-next/livekit/agents/llm/__init__.py) | 152 | Retain | `llm/__init__.py`. Implement the generic LLM contract and fallback; export the realtime and duplex contracts. |
| [llm/_provider_format.py](../../livekit-agents-next/livekit/agents/llm/_provider_format.py) | 142 | Merge | `llm/utils.py`. Same serialization/provider shaping feature as existing utils; no need separate format implementation leaf. |
| [llm/_realtime.py](../../livekit-agents-next/livekit/agents/llm/_realtime.py) | 285 | Retain | `llm/_realtime.py`. Keep concrete OpenAI-compatible implementation separate from generic llm.RealtimeModel. |
| [llm/_realtime_options.py](../../livekit-agents-next/livekit/agents/llm/_realtime_options.py) | 45 | Merge | `llm/_realtime.py`. Small options conversion belongs with concrete provider constructor. |
| [llm/chat_context.py](../../livekit-agents-next/livekit/agents/llm/chat_context.py) | 435 | Retain | `llm/chat_context.py`. Keep the conversation context implementation and existing root/llm public exports. |
| [llm/duplex.py](../../livekit-agents-next/livekit/agents/llm/duplex.py) | 198 | Retain | `llm/duplex.py`. Public duplex model/session namespace; absorbs its adapter. |
| [llm/duplex_adapter.py](../../livekit-agents-next/livekit/agents/llm/duplex_adapter.py) | 87 | Merge | `llm/duplex.py`. Duplex adaptation/gates belong with the duplex contract. |
| [llm/fallback_adapter.py](../../livekit-agents-next/livekit/agents/llm/fallback_adapter.py) | 164 | Merge | `llm/__init__.py`. Standard LLM fallback/stream adapters implement root LLM contract. |
| [llm/mcp.py](../../livekit-agents-next/livekit/agents/llm/mcp.py) | 339 | Retain | `llm/mcp.py`. Keep optional MCP lifecycle/custom SDK stream boundary separate from generic tools. |
| [llm/realtime.py](../../livekit-agents-next/livekit/agents/llm/realtime.py) | 38 | Retain | `llm/realtime.py`. Retain the familiar public generic realtime contract and absorb its fallback implementation. |
| [llm/realtime_fallback_adapter.py](../../livekit-agents-next/livekit/agents/llm/realtime_fallback_adapter.py) | 117 | Merge | `llm/realtime.py`. Group generic realtime fallback with its base contract while preserving llm-root class exports. |
| [llm/remote_chat_context.py](../../livekit-agents-next/livekit/agents/llm/remote_chat_context.py) | 33 | Remove | Not public; its only consumer, Azure Voice Live, moved to the shared Rust realtime path. |
| [llm/tool_context.py](../../livekit-agents-next/livekit/agents/llm/tool_context.py) | 616 | Retain | `llm/tool_context.py`. Single tool authoring/Toolset/decorator implementation with generated tool handling option exports. |
| [llm/utils.py](../../livekit-agents-next/livekit/agents/llm/utils.py) | 472 | Retain | `llm/utils.py`. One provider serialization/schema/argument-format helper leaf. |

### resources

| Current file | Lines | Action | Target and reason |
| --- | ---: | --- | --- |
| [resources/__init__.py](../../livekit-agents-next/livekit/agents/resources/__init__.py) | 1 | Remove marker | Raw assets remain in resources/. Delete package marker while retaining raw resource assets; use the existing livekit.agents package as the importlib.resources anchor. |

### storage

| Current file | Lines | Action | Target and reason |
| --- | ---: | --- | --- |
| [storage/__init__.py](../../livekit-agents-next/livekit/agents/storage/__init__.py) | 44 | Retain | `storage/__init__.py`. Public storage configuration and durable Thread/Lane operations share one file (~430 input lines, plus tiny history projection). |
| [storage/_protocol.py](../../livekit-agents-next/livekit/agents/storage/_protocol.py) | 29 | Split | `_proto.py` + `storage/__init__.py`. `item_to_payload, payload_to_item` → `_proto.py`. `history_of` → `storage/__init__.py`. Move item payload codecs to the shared wire leaf; only the storage row-history projection goes to storage root. |
| [storage/backends.py](../../livekit-agents-next/livekit/agents/storage/backends.py) | 165 | Merge | `storage/__init__.py`. Storage/AgentDBStorage/SQLiteStorage are core package API, not a separate provider abstraction. |
| [storage/context.py](../../livekit-agents-next/livekit/agents/storage/context.py) | 363 | Move | `storage/session.py`. Keep request-scoped A2ASessionContext and native Turn host together as the backend session host component. |
| [storage/context_store.py](../../livekit-agents-next/livekit/agents/storage/context_store.py) | 302 | Move | `storage/context.py`. Application persisted context/transaction authoring becomes the clear storage.context module. |
| [storage/errors.py](../../livekit-agents-next/livekit/agents/storage/errors.py) | 23 | Merge | `storage/__init__.py`. Generated error exports need no handwritten forwarding leaf. |
| [storage/resolve.py](../../livekit-agents-next/livekit/agents/storage/resolve.py) | 192 | Merge | `storage/__init__.py`. Thread/Lane durable handle and resolution APIs belong beside Storage configurations and storage errors. |

### telemetry

| Current file | Lines | Action | Target and reason |
| --- | ---: | --- | --- |
| [telemetry/__init__.py](../../livekit-agents-next/livekit/agents/telemetry/__init__.py) | 57 | Retain | `telemetry/__init__.py`. One import-light telemetry API/context module combines tracer/context, carrier handling, constants, PII helpers and diagnostics. |
| [telemetry/_context.py](../../livekit-agents-next/livekit/agents/telemetry/_context.py) | 216 | Merge | `telemetry/__init__.py`. Tracer and invocation-local context are primary cheap telemetry behavior, not a separate internal concept. |
| [telemetry/_destination.py](../../livekit-agents-next/livekit/agents/telemetry/_destination.py) | 386 | Retain | `telemetry/_destination.py`. Keep per-job authenticated Cloud destination, gating and early-log retention owner. |
| [telemetry/_diagnostics.py](../../livekit-agents-next/livekit/agents/telemetry/_diagnostics.py) | 46 | Merge | `telemetry/__init__.py`. Cheap Counter/lock and suppression ContextVar fit directly in import-light root; file is not needed once root remains exporter-independent. |
| [telemetry/_export.py](../../livekit-agents-next/livekit/agents/telemetry/_export.py) | 291 | Retain | `telemetry/_export.py`. One SDK export boundary owns admission/bounding, snapshots, redaction wrappers and cleanup. |
| [telemetry/_providers.py](../../livekit-agents-next/livekit/agents/telemetry/_providers.py) | 697 | Retain | `telemetry/_providers.py` + `telemetry/__init__.py`. `all except _job_stamp_attributes` → `telemetry/_providers.py`. `_job_stamp_attributes` → `telemetry/__init__.py`. Keep process-lifetime provider adoption, Cloud pipelines and shutdown owner. |
| [telemetry/_redaction.py](../../livekit-agents-next/livekit/agents/telemetry/_redaction.py) | 220 | Merge | `telemetry/_export.py`. SDK record snapshots/redaction and bounded exporter wrappers are the same export boundary. |
| [telemetry/_transport.py](../../livekit-agents-next/livekit/agents/telemetry/_transport.py) | 40 | Merge | `telemetry/__init__.py`. Carrier capture/activation is ordinary invocation-local telemetry API. |
| [telemetry/metrics.py](../../livekit-agents-next/livekit/agents/telemetry/metrics.py) | 153 | Retain | `telemetry/metrics.py`. Keep actual OTel/Prometheus collection independent of the retained public provider metric record module. |
| [telemetry/trace_types.py](../../livekit-agents-next/livekit/agents/telemetry/trace_types.py) | 91 | Merge | `telemetry/__init__.py`. Import-free attribute/event constants can be direct root names rather than another handwritten namespace. |
| [telemetry/traces.py](../../livekit-agents-next/livekit/agents/telemetry/traces.py) | 455 | Retain | `telemetry/traces.py`. Keep session report projection/upload and RTC RPC instrumentation as its real host integration owner. |
| [telemetry/utils.py](../../livekit-agents-next/livekit/agents/telemetry/utils.py) | 106 | Merge | `telemetry/__init__.py`. PII environment lookup, participant attributes and exception recording are cheap core telemetry helpers. |

### tokenize

| Current file | Lines | Action | Target and reason |
| --- | ---: | --- | --- |
| [tokenize/__init__.py](../../livekit-agents-next/livekit/agents/tokenize/__init__.py) | 296 | Retain | `tokenize/__init__.py`. Keep base tokenizer/stream contracts and existing native/basic namespace. |
| [tokenize/blingfire.py](../../livekit-agents-next/livekit/agents/tokenize/blingfire.py) | 83 | Retain | `tokenize/blingfire.py`. Keep baseline concrete tokenizer; eliminating it requires a class naming or abstractness decision beyond file grouping. |

### utils

| Current file | Lines | Action | Target and reason |
| --- | ---: | --- | --- |
| [utils/__init__.py](../../livekit-agents-next/livekit/agents/utils/__init__.py) | 20 | Retain | `utils/__init__.py`. Utility authoring functions, EventEmitter and participant/track waiting helpers form a ~253-line root module. |
| [utils/aio/__init__.py](../../livekit-agents-next/livekit/agents/utils/aio/__init__.py) | 368 | Move | `utils/aio.py`. One async-resource helper module combines channel/task/cancellation primitives and ConnectionPool; no submodules remain to justify an aio package directory. |
| [utils/audio.py](../../livekit-agents-next/livekit/agents/utils/audio.py) | 94 | Move | `utils/media.py`. SDK media helpers combine audio frame/file utilities, native audio decoding and image adaptation. |
| [utils/codecs.py](../../livekit-agents-next/livekit/agents/utils/codecs.py) | 83 | Move | `utils/media.py`. AudioStreamDecoder is audio-only and already used by audio file iteration, so keep it in the same media helper owner. |
| [utils/connection_pool.py](../../livekit-agents-next/livekit/agents/utils/connection_pool.py) | 201 | Move | `utils/aio.py`. Generic asynchronous pool reuse/TTL/resource closure belongs with async resource helpers. |
| [utils/http_context.py](../../livekit-agents-next/livekit/agents/utils/http_context.py) | 367 | Retain | `utils/http_context.py`. Keep aiohttp request/session/SSL/WebSocket resource owner independent from generic pool primitives and ASGI server. |
| [utils/images.py](../../livekit-agents-next/livekit/agents/utils/images.py) | 38 | Move | `utils/media.py`. Image encoder/frame adapter and audio codec adapters are one small SDK-media boundary, with clear audio and image sections. |
| [utils/misc.py](../../livekit-agents-next/livekit/agents/utils/misc.py) | 97 | Merge | `utils/__init__.py`. Tiny miscellaneous functions/decorator form the utility root API instead of another import hop. |
| [utils/participant.py](../../livekit-agents-next/livekit/agents/utils/participant.py) | 136 | Merge | `utils/__init__.py`. Participant/track waits are simple public utilities and fit in the root alongside existing helpers. |

### voice

| Current file | Lines | Action | Target and reason |
| --- | ---: | --- | --- |
| [voice/__init__.py](../../livekit-agents-next/livekit/agents/voice/__init__.py) | 52 | Retain | `voice/__init__.py`. Real public voice namespace implements Agent authoring and exports remaining session/feature modules. |
| [voice/_session_bindings.py](../../livekit-agents-next/livekit/agents/voice/_session_bindings.py) | 541 | Retain | `voice/_session_bindings.py`. One native session construction/configuration/host-access boundary. |
| [voice/_session_callbacks.py](../../livekit-agents-next/livekit/agents/voice/_session_callbacks.py) | 329 | Retain | `voice/_session_callbacks.py`. One bidirectional session host callback adapter: hooks/node iterators and native event delivery. |
| [voice/_session_events.py](../../livekit-agents-next/livekit/agents/voice/_session_events.py) | 348 | Merge | `voice/_session_callbacks.py`. Native event projection and application hook adaptation share session host boundary. |
| [voice/_session_host.py](../../livekit-agents-next/livekit/agents/voice/_session_host.py) | 171 | Merge | `voice/_session_bindings.py`. SessionHost startup assembly already reaches into binding helper implementation. |
| [voice/_session_types.py](../../livekit-agents-next/livekit/agents/voice/_session_types.py) | 82 | Merge | `voice/_session_bindings.py`. Public node/callback signatures, sampler and option adaptation sit with their native binding consumer. |
| [voice/agent.py](../../livekit-agents-next/livekit/agents/voice/agent.py) | 732 | Merge | `voice/__init__.py`. Agent configuration/authoring and its private local-tool descriptor are one public namespace implementation. |
| [voice/agent_host.py](../../livekit-agents-next/livekit/agents/voice/agent_host.py) | 528 | Merge | `voice/agent_run.py`. Shared run validation/callback execution belongs with invocation binding/driver. |
| [voice/agent_run.py](../../livekit-agents-next/livekit/agents/voice/agent_run.py) | 340 | Retain | `voice/agent_run.py`. One host invocation implementation: run binding, driver, AgentCallbacks, validation and tool templates. |
| [voice/agent_session.py](../../livekit-agents-next/livekit/agents/voice/agent_session.py) | 539 | Retain | `voice/agent_session.py`. Public VoiceSession/VoiceTask lifecycle and command surface. |
| [voice/agent_tool.py](../../livekit-agents-next/livekit/agents/voice/agent_tool.py) | 73 | Merge | `voice/__init__.py`. Private descriptor belongs with Agent.as_tool authoring. |
| [voice/amd.py](../../livekit-agents-next/livekit/agents/voice/amd.py) | 268 | Retain | `voice/amd.py`. Keep answering machine detection as coherent inference/job integration feature. |
| [voice/avatar.py](../../livekit-agents-next/livekit/agents/voice/avatar.py) | 499 | Retain | `voice/avatar.py`. Keep documented avatar provider/RTC media transport namespace. |
| [voice/background_audio.py](../../livekit-agents-next/livekit/agents/voice/background_audio.py) | 282 | Retain | `voice/background_audio.py`. Keep background asset/audio playback ownership separate from general RTC interfaces. |
| [voice/events.py](../../livekit-agents-next/livekit/agents/voice/events.py) | 364 | Retain | `voice/events.py`. Keep documented voice.events event namespace and real RunContext implementation. |
| [voice/io.py](../../livekit-agents-next/livekit/agents/voice/io.py) | 405 | Retain | `voice/io.py`. One host protocol/handle module for media input/output, detector stream ports and speech playout. |
| [voice/remote_session.py](../../livekit-agents-next/livekit/agents/voice/remote_session.py) | 120 | Merge | `voice/_session_bindings.py`. Native remote protocol host access and ConsoleCall claim state belong to session host setup. |
| [voice/room_io.py](../../livekit-agents-next/livekit/agents/voice/room_io.py) | 630 | Retain | `voice/room_io.py`. Keep concrete livekit.rtc SDK adapter separate from interfaces in io. |
| [voice/run_result.py](../../livekit-agents-next/livekit/agents/voice/run_result.py) | 602 | Retain | `voice/run_result.py`. One invocation result consumption module: settled results, assertions, captured voice runs and stream handle. |
| [voice/run_stream.py](../../livekit-agents-next/livekit/agents/voice/run_stream.py) | 187 | Merge | `voice/run_result.py`. Streaming and settled result handles share user-visible invocation consumption lifecycle. |
| [voice/speech_handle.py](../../livekit-agents-next/livekit/agents/voice/speech_handle.py) | 159 | Merge | `voice/io.py`. SpeechHandle is output/playout host handle, fits IO ownership alongside AudioOutput. |
| [voice/tool_executor.py](../../livekit-agents-next/livekit/agents/voice/tool_executor.py) | 64 | Split | `voice/agent_run.py` + `llm/tool_context.py`. Rendering → agent_run; option/prompt exports → tool_context. Template rendering and default UPDATE_TEMPLATE join invocation execution; public option/prompt exports join tools. |
| [voice/turn.py](../../livekit-agents-next/livekit/agents/voice/turn.py) | 86 | Merge | `voice/io.py`. Turn detector protocols/mode/options are media pipeline ports, not independent execution machinery. |
| [voice/voice_task.py](../../livekit-agents-next/livekit/agents/voice/voice_task.py) | 44 | Merge | `voice/agent_session.py`. Small live-session temporary interaction wrapper belongs with VoiceSession. |

