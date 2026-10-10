"""CRM payload adapters for :class:`~.models.PostCallReport`.

These are pure payload builders: no network calls, no CRM SDK dependencies.
Users POST the returned dicts to the CRM APIs with their own authenticated
client (e.g. a HubSpot private-app token or a Salesforce connected app).
"""

from __future__ import annotations

import time
from typing import Any

from .models import PostCallReport


def _format_transcript(report: PostCallReport) -> str:
    """Render the transcript turns plus a tool-invocation summary as plain text."""
    lines: list[str] = []
    for turn in report.turns:
        marker = " [interrupted]" if turn.interrupted else ""
        lines.append(f"{turn.speaker.capitalize()}: {turn.text}{marker}")

    if report.tool_invocations:
        lines.append("")
        lines.append("Tool invocations:")
        for rec in report.tool_invocations:
            duration = f"{rec.duration_ms:.0f}ms" if rec.duration_ms is not None else "untimed"
            detail = rec.error if rec.error is not None else rec.result
            suffix = f" — {detail}" if detail else ""
            lines.append(f"- {rec.tool_name} ({rec.status}, {duration}){suffix}")

    return "\n".join(lines)


def to_hubspot_engagement(report: PostCallReport) -> dict[str, Any]:
    """Build a HubSpot v3 calls-engagement payload from a post-call report.

    Payload builder only — POST it to ``/crm/v3/objects/calls`` with your own
    authenticated HubSpot client. ``hs_call_duration`` is expressed in
    milliseconds as a string and ``hs_timestamp`` in epoch milliseconds, per
    HubSpot's engagement conventions.
    """
    title = f"Call: {report.room_name or report.report_id}"
    return {
        "properties": {
            "hs_call_title": title,
            "hs_call_body": _format_transcript(report),
            "hs_call_duration": str(int(report.metrics.total_duration_seconds * 1000)),
            "hs_call_direction": "INBOUND",
            "hs_call_status": "COMPLETED",
            "hs_timestamp": int(report.created_at * 1000),
        }
    }


def to_salesforce_task(report: PostCallReport) -> dict[str, Any]:
    """Build a Salesforce Task (ActivityHistory) payload from a post-call report.

    Payload builder only — POST it to ``/services/data/vXX.X/sobjects/Task``
    with your own authenticated Salesforce client.
    """
    return {
        "Subject": f"Call: {report.room_name or report.report_id}",
        "Description": _format_transcript(report),
        "Status": "Completed",
        "TaskSubtype": "Call",
        "CallType": "Inbound",
        "CallDurationInSeconds": int(report.metrics.total_duration_seconds),
        "ActivityDate": time.strftime("%Y-%m-%d", time.gmtime(report.created_at)),
    }
