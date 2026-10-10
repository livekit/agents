#!/usr/bin/env python3
"""Demonstrate xAI RealtimeModel websocket recycle via max_session_duration."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import pass_fail, require_xai_api_key, run_async  # noqa: E402


def _print_construction_notes(seconds: float) -> None:
    print("Construction without recycle (plugin default):")
    print("  from livekit.plugins.xai.realtime import RealtimeModel")
    print("  model = RealtimeModel()  # max_session_duration=None")
    print()
    print("Construction with production-like 20m recycle:")
    print("  model = RealtimeModel(max_session_duration=20 * 60)")
    print()
    print(f"This demo uses max_session_duration={seconds} so reconnect is observable locally.")
    print("Watch for: reconnecting / reconnected log lines, and session_reconnected.")


async def _run_live(seconds: float) -> int:
    require_xai_api_key()

    from livekit.plugins.xai.realtime import RealtimeModel

    _print_construction_notes(seconds)

    model = RealtimeModel(max_session_duration=seconds)
    session = model.session()
    reconnected = asyncio.Event()

    def _on_reconnected(_: object) -> None:
        print("event: session_reconnected")
        reconnected.set()

    session.on("session_reconnected", _on_reconnected)

    print(f"recycle timer armed for {seconds}s; watching for session_reconnected")
    try:
        # give the websocket a moment to come up before the recycle sleep starts counting
        await asyncio.sleep(2.0)
        try:
            await asyncio.wait_for(reconnected.wait(), timeout=seconds + 30.0)
        except TimeoutError:
            return pass_fail(
                False,
                f"session_reconnected did not fire within {seconds + 30:.0f}s",
            )
        return pass_fail(True, f"session_reconnected after ~{seconds}s max_session_duration")
    finally:
        await session.aclose()
        await model.aclose()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-log",
        action="store_true",
        help="Print construction notes only; do not connect",
    )
    args, _unknown = parser.parse_known_args()
    seconds = float(os.environ.get("XAI_RECYCLE_SECONDS", "45"))
    dry = args.dry_log or os.environ.get("XAI_RECYCLE_DRY", "") == "1"

    if dry:
        _print_construction_notes(seconds)
        return pass_fail(True, "dry-log only; no websocket opened")

    return await _run_live(seconds)


if __name__ == "__main__":
    run_async(main)
