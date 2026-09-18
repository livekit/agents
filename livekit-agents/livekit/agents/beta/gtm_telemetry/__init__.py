"""Post-call GTM/CRM telemetry — models, collector, webhook dispatcher, and CRM adapters.

Usage::

    from livekit.agents.beta.gtm_telemetry import (
        PostCallTelemetryCollector,
        PostCallReport,
        WebhookDispatcher,
        to_salesforce_task,
        to_hubspot_engagement,
    )
"""

from .adapters import to_hubspot_engagement, to_salesforce_task
from .collector import PostCallTelemetryCollector
from .models import CallMetrics, PostCallReport, ToolInvocationRecord, TranscriptTurn
from .webhook import WebhookDispatcher

__all__ = [
    # collector
    "PostCallTelemetryCollector",
    # webhook
    "WebhookDispatcher",
    # models
    "PostCallReport",
    "ToolInvocationRecord",
    "TranscriptTurn",
    "CallMetrics",
    # adapters
    "to_hubspot_engagement",
    "to_salesforce_task",
]
