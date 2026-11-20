"""Check one audio file for a TTS provider's AI-generation signal.

Usage (from examples/inference):
    uv run watermark/check_file.py reply.wav --provider cartesia
    uv run watermark/check_file.py interview.mp3 --provider inworld

Needs LIVEKIT_API_KEY and LIVEKIT_API_SECRET (or LIVEKIT_INFERENCE_API_KEY /
LIVEKIT_INFERENCE_API_SECRET) in the environment or in examples/.env, for a
project where LiveKit has enabled watermark checks.

Exit codes: 0 the check ran (whatever the verdict), 1 the check was refused or
failed, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import APIConnectionError, APIStatusError, inference, utils

load_dotenv(Path(__file__).parents[2] / ".env")


def describe(result: inference.WatermarkResult) -> str:
    """One line per result. `type` and `outcome` are open strings: pass unknown values through."""
    if result.outcome is not None:
        return f"{result.type}: {result.outcome}"
    if result.probability is not None:
        return f"{result.type}: probability {result.probability:.2f} (no threshold applied)"
    return f"{result.type}: the provider could not score this file"


def describe_refusal(error: APIStatusError) -> str:
    # body is the gateway's {"type", "error", "hint"} dict, or plain text for a
    # 401, a suspended project's 403 and a draining gateway's 503.
    body = error.body
    if isinstance(body, dict):
        hint = f" ({body['hint']})" if body.get("hint") else ""
        return f"{error.status_code} {body.get('type')}: {body.get('error')}{hint}"
    return f"{error.status_code}: {body}"


async def check(path: Path, provider: str) -> int:
    # Outside an agent job, open an HTTP context, or pass http_session= to the detector.
    async with utils.http_context.open():
        detector = inference.WatermarkDetector()
        try:
            check = await detector.check(path, provider=provider)
        except APIStatusError as e:
            print(f"refused: {describe_refusal(e)}", file=sys.stderr)
            return 1
        except APIConnectionError as e:
            # APITimeoutError is a subclass: the attempt connected but took over 190 s.
            print(f"gateway unreachable: {e}", file=sys.stderr)
            return 1

    print(f"provider: {check.provider}  request_id: {check.request_id}")
    for result in check.results:
        print(f"  {describe(result)}")
    # not_detected only means this provider's signal is absent from this file,
    # not that the audio is human speech.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path, help="WAV, MP3, OGG, FLAC, M4A or WebM; not raw PCM")
    parser.add_argument("--provider", required=True, choices=["cartesia", "inworld"])
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error(f"{args.file} is not a file")
    return asyncio.run(check(args.file, args.provider))


if __name__ == "__main__":
    sys.exit(main())
