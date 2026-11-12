# Contributor guide

These pages explain how the framework is implemented. The [public API docs](../README.md)
describe application behavior; [`AGENTS.md`](../../AGENTS.md) owns the repository rules.

Start with the [execution paths](architecture.md#execution-paths) to follow a Python call into its Rust owner.
Use [development](development.md) for installation, builds, checks and review conventions.

## Architecture and execution

| Guide | Use it for |
| --- | --- |
| [Architecture](architecture.md) | Crate dependencies, sans-I/O decisions, effects and runtime composition |
| [Component contracts](components.md) | Ownership of agent execution, text routing, voice, providers, room I/O, storage and workers |
| [Bindings and Python](bindings.md) | ABI declarations, code generation, facades, host values, GIL boundaries and plugins |
| [Runtime contracts](runtime.md) | Calls, streams, cancellation, resource lifetimes, backpressure, retries and managed turns |
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
| [DST contribution guide](../../rust/crates/lk-agents-fuzz/CONTRIBUTING.md) | Add a regression, model a fake port, reproduce a seed and diagnose a failure |

Keep current contracts in their owning guide. Migration ledgers, dated file counts and
completed audit checklists are not part of the implementation reference.
