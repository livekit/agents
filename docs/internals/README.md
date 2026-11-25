# Contributor guide

These pages explain how the framework is implemented. The [public API docs](../README.md)
describe application behavior; [`AGENTS.md`](../../AGENTS.md) owns the repository rules.

Start with the [execution paths](architecture.md#execution-paths) to follow a Python call into its Rust owner.
Use [development](development.md) for installation, builds, checks and review conventions.

## Architecture and execution

| Guide | Use it for |
| --- | --- |
| [Architecture](architecture.md) | Execution paths, crate dependencies, source layout and machine composition |
| [Component contracts](components.md) | Decision ownership and component-specific behavior for agents, sessions, providers, rooms, storage and workers |
| [Bindings and Python](bindings.md) | Host value semantics, ABI declarations, code generation, facades and the GIL boundary |
| [Runtime contracts](runtime.md) | Shared call/stream lifetimes, cancellation, resource ownership, backpressure and effect execution |
| [Processes and IPC](processes.md) | Job supervision, fork safety, preloading, shutdown and process logging |

## Persistence and observability

| Guide | Use it for |
| --- | --- |
| [Storage](storage.md) | SQLite and AgentDB, partitions, leases, fenced checkpoints, history and the journal |
| [Telemetry internals](telemetry.md) | Signal production, spans, metrics, context propagation, profiles and export |
| [Telemetry reference](telemetry-reference.md) | Emitted fields, span trees, recording uploads and Cloud ingestion |

## Development and verification

| Guide | Use it for |
| --- | --- |
| [Development and review](development.md) | Local builds, checks, source conventions and review evidence |
| [Internal testing](testing.md) | Component, boundary and deterministic simulation tests |
| [DST guide](../../rust/crates/lk-agents-fuzz/README.md) | Add a regression, model a fake port, reproduce a seed and diagnose a failure |

## Design proposals

[Shared RoomIO](../proposals/room-io.md) is the alignment proposal for room bindings
shared by voice, translation, transcription and application-defined sessions.
It covers shared ownership and bindings with existing legacy protocols; a new
Protobuf session protocol is deferred and TBD. Its proposed API and unresolved
decisions are separate from current contracts.

[Multi-turn AMD](../proposals/amd.md) proposes answering-machine detection as a machine that
gates each reply through effects and turn-keyed answers, ported from the Python
implementation's spec.

Keep current contracts in their owning guide. Migration ledgers, dated file counts and
completed audit checklists are not part of the implementation reference.
Architecture links to component decisions instead of defining them again. Binding
adapters refer to runtime lifetimes; component guides describe only their domain's
additional rules. Storage, telemetry and testing own their detailed mechanics.
