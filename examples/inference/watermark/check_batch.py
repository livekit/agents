"""Check every audio file in a directory and write one JSON line per file.

Usage (from examples/inference):
    uv run watermark/check_batch.py recordings/ --provider cartesia > results.jsonl

Each check counts against the project's daily allowance (reset 00:00 UTC), so
the batch runs a few checks at a time and stops at the first refusal that will
not clear on its own: an auth or enablement error, or a spent allowance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from livekit.agents import APIConnectionError, APIStatusError, inference, utils

load_dotenv(Path(__file__).parents[2] / ".env")

AUDIO_SUFFIXES = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
# The gateway also caps uploads in progress per project; stay well under it.
CONCURRENCY = 4


class StopBatch(Exception):
    """A refusal every later file would get too."""


def is_terminal(error: APIStatusError) -> bool:
    # 401/403: credentials, suspension, not enabled, HIPAA or region.
    # 429 that reaches the caller: the daily allowance, or a wait over 60 s.
    return error.status_code in (401, 403, 429)


def record(path: Path, **fields: Any) -> str:
    return json.dumps({"file": str(path), **fields})


async def check_one(
    detector: inference.WatermarkDetector, path: Path, provider: str, gate: asyncio.Semaphore
) -> str:
    async with gate:
        try:
            check = await detector.check(path, provider=provider)
        except APIStatusError as e:
            if is_terminal(e):
                raise StopBatch(f"{e.status_code}: {e.body}") from e
            return record(path, error=e.status_code, body=e.body)
        except APIConnectionError as e:
            return record(path, error="connection", detail=str(e))
    results = [
        {"type": r.type, "outcome": r.outcome, "probability": r.probability} for r in check.results
    ]
    return record(path, request_id=check.request_id, results=results)


async def run(directory: Path, provider: str) -> int:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    gate = asyncio.Semaphore(CONCURRENCY)
    async with utils.http_context.open():
        detector = inference.WatermarkDetector()
        tasks = [asyncio.create_task(check_one(detector, p, provider, gate)) for p in files]
        try:
            for task in asyncio.as_completed(tasks):
                print(await task, flush=True)
        except StopBatch as e:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            print(f"stopped: {e}", file=sys.stderr)
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--provider", required=True, choices=["cartesia", "inworld"])
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error(f"{args.directory} is not a directory")
    return asyncio.run(run(args.directory, args.provider))


if __name__ == "__main__":
    sys.exit(main())
