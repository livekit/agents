"""Post-hoc run artifacts for the hotel-receptionist simulations.

When ``LIVEKIT_SESSION_REPORT_DIR`` is set, each job
dumps two files there, named per-room so concurrent jobs don't collide:

This is deliberately kept out of agent.py and the SDK; it's pure plumbing for the
benchmark harness, not part of the agent's behavior.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hotel_db import HotelDB

    from livekit.agents import JobContext
    from livekit.agents.voice.report import SessionReport

logger = logging.getLogger("hotel-receptionist")


def dump_run_artifacts(ctx: JobContext, report: SessionReport, db: HotelDB) -> None:
    """Write the final DB and session report to LIVEKIT_SESSION_REPORT_DIR (no-op if unset)."""
    report_dir = os.environ.get("LIVEKIT_SESSION_REPORT_DIR")
    if not report_dir:
        return

    room = ctx.room.name
    try:
        with open(os.path.join(report_dir, f"final_db-{room}.sqlite"), "wb") as f:
            f.write(db.serialize())
    except Exception:
        logger.exception("error dumping final DB state")

    try:
        report_dict = report.to_dict()
        report_dict["tags"] = sorted(ctx.tagger.tags)
        report_dict["evaluations"] = ctx.tagger.evaluations
        report_dict["outcome"] = ctx.tagger.outcome
        report_dict["outcome_reason"] = ctx.tagger.outcome_reason
        with open(os.path.join(report_dir, f"session_report-{room}.json"), "w") as f:
            json.dump(report_dict, f, indent=2)
    except Exception:
        logger.exception("error dumping session report")
