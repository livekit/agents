from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
    from .report import SessionReport


@dataclass(frozen=True)
class RecordingExportResult:
    """References produced by a recording exporter.

    These values may be attached to the ``agent_session`` span. Prefer governed
    artifact ids or signed, expiring URLs over raw storage paths for sensitive
    recordings.
    """

    recording_url: str | None = None
    """Externally resolvable recording URL, ideally signed or access-controlled."""

    recording_id: str | None = None
    """Opaque recording artifact id suitable for joining traces to storage."""

    recording_path: str | Path | None = None
    """Optional local/storage path. Only set this when it is safe to expose in traces."""

    trace_attributes: Mapping[str, AttributeValue] = field(default_factory=dict)
    """Additional OpenTelemetry attributes to attach to the session span."""


class RecordingExporter(ABC):
    """Exports a completed session recording to an external backend."""

    @abstractmethod
    async def export(self, report: SessionReport) -> RecordingExportResult | None:
        """Export the completed recording and return trace-linking metadata."""
