# Telemetry integration

Configure application providers before starting jobs. The framework adopts the current
tracer provider at setup and adds its Cloud destination without replacing the global
provider, sampler, resource or application exporters. Job close flushes framework work;
process exit closes framework-owned resources.

For an explicit provider, select it with `set_tracer_provider(provider)`. Instrument
child work through `trace.get_tracer(name, tracer_provider=provider)`. HostInvoker enters
the invocation's captured context around callbacks and restores it on return or cancel.

Wrap each application exporter in `RedactingSpanExporter(exporter)` or
`RedactingLogExporter(exporter)` before adding its SDK processor. Each destination gets
its own snapshot. `allow_pii=True` also requires content capture opt-in. Set
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` to opt in. Consent is captured at
creation; records without captured consent stay redacted. Project redaction
withholds raw audio and transcripts. Providers already configured with exporters must
add their own destination wrappers; a provider-wide redaction request reports a conflict.

An existing application meter provider keeps its readers. Cloud metrics require a reader
registered at provider construction; an incompatible existing provider reports a conflict.
Cloud tracing and logs remain independently available.

Framework SDK queues are bounded by record count, and each record's estimated bytes are
admitted before batching. Flush and shutdown wait a bounded time; a callback still running
keeps its own thread and counts as incomplete cleanup. Read `telemetry.diagnostics()` for
nonrecursive drop/failure/conflict/cleanup counters.
Destination wrappers reject oversized records with an exporter failure and an observable drop.

Session-report events retain at most 1,024 snapshots/1 MiB, with 64 KiB per event
and one-hour expiry on access. Session creation captures content consent; project
redaction narrows it. Events containing live objects are omitted from reports.
A report for a session started in a job is readable only by that job.
Console report JSON and Cloud uploads share native projection and destination redaction.
Safe state/usage events remain available; withheld history/options keep explicit empty fields.
Audio requires its explicit recording option; project redaction excludes raw audio and transcript.
Explicit `SessionReport` objects capture the current content setting when constructed.
Job-end report creation and metric folding run off the event loop; console writes and the
Cloud recording upload run natively, each under its own small process admission.
`JobContext.make_session_report()` returns a synchronous detached snapshot.

Run `uv run pytest tests/test_telemetry_destinations.py --unit` for destination isolation,
capture consent and provider behavior. Run `cargo test -p lk-agents-runtime telemetry`
and `cargo test -p lk-agents-net session_report` from `rust/` for supplied-time policy,
event retention, streaming uploads and the transcript bounds.
The Python-feature runtime tests additionally check actual OTel parent IDs through
HostInvoker, including failures and cancellation.
