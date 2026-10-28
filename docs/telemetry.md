# Telemetry

The framework adopts the application's configured OpenTelemetry tracer
provider. Configure it before starting jobs, or select an explicit provider
with `livekit.agents.telemetry.set_tracer_provider(provider)`. The application's
global provider, sampler, resource and exporters remain application-owned.
Framework cleanup does not shut down an application-owned provider.

The framework records its spans, metrics and logs through the same SDK, so the
application's processors and exporters receive them alongside its own
instrumentation.

For child instrumentation with an explicit provider, use
`trace.get_tracer(name, tracer_provider=provider)`. Tools and callbacks inherit
their invocation's tracing context, including across failure and cancellation.

## Export destinations

Wrap each application exporter before adding its SDK processor:

| `livekit.agents.telemetry` | Purpose |
| --- | --- |
| `RedactingSpanExporter(exporter, allow_pii=False)` | Apply content policy to one span destination |
| `RedactingLogExporter(exporter, allow_pii=False)` | Apply content policy to one log destination |
| `diagnostics()` | Read drop, export-failure and cleanup counters as `dict[str, int]` |

Each destination receives an independent snapshot. Filtering one destination
cannot change another's data. `allow_pii=True` also requires capture consent
when the record is created; set
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` to opt in. Enabling
capture later does not reveal records created without consent. Project
redaction narrows content access further.

Attach wrappers to existing application exporters explicitly. Provider-wide
filtering cannot establish a separate policy for each destination; conflicting
configuration is reported. An existing meter provider keeps its readers;
Cloud metrics require a compatible reader configured at provider construction.

Export queues are bounded by record count and retained bytes. Saturation and
cleanup failures are observable through diagnostics without blocking the
conversation. See the [runtime telemetry contract](internals/rust-core.md#telemetry)
for ownership, limits and verification.
