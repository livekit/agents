"""Post-call CRM & analytics telemetry: report collector, signed webhook delivery, and
pure CRM payload adapters.

Beta: this package's API and schema are not covered by semver stability guarantees and
may change in a future release. Opt-in only — nothing here runs unless you construct
and attach a :class:`PostCallTelemetryCollector` yourself.

Transcripts and tool arguments/results may contain personal or confidential
information. You control whether collection is enabled, what it includes
(``CollectorConfig``), and whether/where it is delivered. Webhook destinations are
trusted by this library without further validation — treat them accordingly. This
module performs no automatic redaction; use ``PostCallTelemetryCollector(redact=...)``
if you need one.

Example::

    from livekit.agents.beta.gtm_telemetry import (
        PostCallTelemetryCollector,
        PostCallWebhookDispatcher,
        WebhookConfig,
    )

    collector = PostCallTelemetryCollector(metadata={"lead_id": "lead_123"})
    collector.attach(session)

    await session.start(...)

    report = collector.finalize()

    dispatcher = PostCallWebhookDispatcher(
        WebhookConfig(url=os.environ["POST_CALL_WEBHOOK_URL"],
                       secret=os.environ["POST_CALL_WEBHOOK_SECRET"])
    )
    try:
        await dispatcher.send(report)
    finally:
        await dispatcher.aclose()
"""

from .adapters import build_hubspot_engagement, build_salesforce_task
from .collector import PostCallTelemetryCollector
from .models import (
    CollectorConfig,
    JsonValue,
    MetricAggregate,
    PostCallReport,
    SessionMetricsSummary,
    ToolExecutionRecord,
    ToolProgressUpdate,
    TranscriptTurn,
)
from .webhook import PostCallWebhookDispatcher, WebhookConfig, WebhookDeliveryError

__all__ = [
    "PostCallTelemetryCollector",
    "CollectorConfig",
    "PostCallReport",
    "TranscriptTurn",
    "ToolExecutionRecord",
    "ToolProgressUpdate",
    "SessionMetricsSummary",
    "MetricAggregate",
    "JsonValue",
    "WebhookConfig",
    "PostCallWebhookDispatcher",
    "WebhookDeliveryError",
    "build_hubspot_engagement",
    "build_salesforce_task",
]
