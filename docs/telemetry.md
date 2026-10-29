# Telemetry

Read consumption directly from `result.usage`, `stream.usage`, or
`session.usage`; no exporter or event subscription is required.
[Usage and metrics](usage.md) defines accounting records, scopes, and measurement
availability. This page defines how observations reach telemetry destinations.

The framework adopts the application's configured OpenTelemetry tracer
provider. Configure it before starting jobs, or select an explicit provider
with `livekit.agents.telemetry.set_tracer_provider(provider)`. The application's
global provider, sampler, resource and exporters remain application-owned.
Framework cleanup does not shut down an application-owned provider.

```python
set_tracer_provider(
    provider,                          # the provider framework spans use; the OTel global is unchanged
    metadata={"deployment": "eu-1"},   # attributes stamped on every span the provider starts
    allow_pii=None,                    # see "Content and PII" below
)
```

`metadata=` applies to an SDK `TracerProvider`; a later call with new metadata
replaces the previous values for that provider. Attributes a span already
carries win over it.

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
| `diagnostics()` | Read the process's export counters as `dict[str, int]`: `dropped`, `failed`, `conflicts` (configuration conflicts reported) and `incomplete_cleanup` |

Each destination receives an independent snapshot. Filtering one destination
cannot change another's data. Project redaction narrows content access further.

## Content and PII

| `livekit.agents.telemetry` | Purpose |
| --- | --- |
| `set_capture_content(enabled)` | Capture consent: whether records carry message content |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` | The same consent from the environment, read at import |
| `LIVEKIT_TELEMETRY_ALLOW_PII` | `set_tracer_provider(allow_pii=)` when the call omits it, or when the framework adopts the ambient provider |

Capture consent is off by default. A destination wrapper with `allow_pii=True`
exports content only from records created with consent: enabling capture later
does not reveal records created without it.

`set_tracer_provider(allow_pii=False)`, or `LIVEKIT_TELEMETRY_ALLOW_PII=0`
(also `false`, `no`, `off`), asks for provider-wide filtering. The framework
cannot filter application processors, so it warns once per provider and leaves
filtering to the destination wrappers above. Any other value, or `allow_pii=True`,
leaves each destination's own policy in place.

Attach wrappers to existing application exporters explicitly. Provider-wide
filtering cannot establish a separate policy for each destination; conflicting
configuration is reported. An existing meter provider keeps its readers;
Cloud metrics require a compatible reader configured at provider construction.

Export queues are bounded by record count and retained bytes. Saturation and
cleanup failures are observable through diagnostics without blocking the
conversation. See the [runtime telemetry contract](internals/rust-core.md#telemetry)
for ownership, limits and verification.
