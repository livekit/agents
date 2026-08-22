"""Pure CRM payload builders.

Beta: not covered by semver stability guarantees. External CRM API schemas change over
time — these builders are a generic starting point, not full API coverage; consult the
target CRM's current API docs before wiring a payload to a real portal.

None of these functions perform network calls, require a CRM SDK, read environment
variables, or mutate the input :class:`~.models.PostCallReport`.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import JsonValue, PostCallReport, to_json_safe


def _format_transcript(report: PostCallReport, *, max_chars: int | None) -> tuple[str, bool, int]:
    full_text = "\n".join(f"{turn.role}: {turn.text}" for turn in report.transcript if turn.text)
    if max_chars is None or len(full_text) <= max_chars:
        return full_text, False, 0
    return full_text[:max_chars], True, len(full_text) - max_chars


def _tool_summary(report: PostCallReport) -> str:
    return ", ".join(
        f"{record.name} ({'error' if record.is_error else record.status})"
        for record in report.tool_executions
    )


def _copy_metadata(report: PostCallReport) -> dict[str, JsonValue]:
    # to_json_safe rebuilds every nested container fresh, giving an independent copy
    # that shares no mutable state with report.metadata
    copied = to_json_safe(dict(report.metadata))
    assert isinstance(copied, dict)
    return copied


def build_hubspot_engagement(
    report: PostCallReport,
    *,
    owner_id: str | None = None,
    contact_ids: Sequence[str] | None = None,
    title: str | None = None,
    transcript_max_chars: int | None = 10_000,
) -> dict[str, JsonValue]:
    """Build a JSON-safe payload shaped for a HubSpot call Engagement.

    Args:
        report: The report to summarize.
        owner_id: Caller-supplied HubSpot owner id. Never fabricated.
        contact_ids: Caller-supplied HubSpot contact ids to associate. Never fabricated.
        title: Engagement title; defaults to a generic "Post-call report".
        transcript_max_chars: Truncate the transcript body to this many characters. When
            truncated, ``metadata.transcriptTruncated``/``transcriptOmittedChars`` in the
            returned payload record that fact — content is never silently dropped
            without a record of it. ``None`` disables truncation.
    """
    transcript, truncated, omitted_chars = _format_transcript(
        report, max_chars=transcript_max_chars
    )
    metadata = _copy_metadata(report)

    return {
        "engagement": {
            "type": "CALL",
            "title": title or "Post-call report",
            "timestamp": report.started_at,
        },
        "associations": {
            "ownerId": owner_id,
            "contactIds": list(contact_ids) if contact_ids else [],
        },
        "metadata": {
            "durationMilliseconds": (
                int(report.duration * 1000) if report.duration is not None else None
            ),
            "body": transcript,
            "transcriptTruncated": truncated,
            "transcriptOmittedChars": omitted_chars,
            "toolSummary": _tool_summary(report),
            "disposition": metadata.get("disposition"),
        },
        "source": {
            "jobId": report.job_id,
            "roomName": report.room_name,
            "endReason": report.end_reason,
            "reportId": report.report_id,
        },
        "customProperties": metadata,
    }


def build_salesforce_task(
    report: PostCallReport,
    *,
    who_id: str | None = None,
    what_id: str | None = None,
    owner_id: str | None = None,
    subject: str | None = None,
    transcript_max_chars: int | None = 10_000,
) -> dict[str, JsonValue]:
    """Build a JSON-safe payload shaped for a Salesforce Task/Activity.

    Args:
        report: The report to summarize.
        who_id: Caller-supplied Salesforce Contact/Lead id. Never fabricated.
        what_id: Caller-supplied Salesforce related-object id. Never fabricated.
        owner_id: Caller-supplied Salesforce owner (User) id. Never fabricated.
        subject: Task subject; defaults to a generic "Post-call report".
        transcript_max_chars: Truncate the description to this many characters. When
            truncated, ``descriptionTruncated``/``descriptionOmittedChars`` in the
            returned payload record that fact. ``None`` disables truncation.

    Not tied to any specific Salesforce API version.
    """
    transcript, truncated, omitted_chars = _format_transcript(
        report, max_chars=transcript_max_chars
    )
    metadata = _copy_metadata(report)

    return {
        "WhoId": who_id,
        "WhatId": what_id,
        "OwnerId": owner_id,
        "Subject": subject or "Post-call report",
        "Status": "Completed",
        "Type": "Call",
        "CallDurationInSeconds": report.duration,
        "ActivityDate": report.started_at,
        "Description": transcript,
        "descriptionTruncated": truncated,
        "descriptionOmittedChars": omitted_chars,
        "toolSummary": _tool_summary(report),
        "disposition": metadata.get("disposition"),
        "source": {
            "jobId": report.job_id,
            "roomName": report.room_name,
            "endReason": report.end_reason,
            "reportId": report.report_id,
        },
        "customFields": metadata,
    }
