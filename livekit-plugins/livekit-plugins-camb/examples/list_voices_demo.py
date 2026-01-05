#!/usr/bin/env python3
"""
Quick script to list available Camb.ai voices.

Usage:
    cd livekit-plugins/livekit-plugins-camb
    uv run python examples/list_voices_demo.py
"""

import asyncio
import os
import sys

# Load .env file from examples directory
from pathlib import Path

try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not required, can use export instead

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from livekit.plugins.camb import list_voices


async def main():
    if not os.getenv("CAMB_API_KEY"):
        print("Error: CAMB_API_KEY environment variable not set")
        return

    print("Fetching available voices from Camb.ai...\n")

    try:
        voices = await list_voices()
        print(f"✅ Found {len(voices)} voices\n")
        print(f"{'ID':<10} {'Name':<40} {'Gender':<15} {'Language'}")
        print("-" * 80)

        for voice in voices:
            print(
                f"{voice.id:<10} {voice.name:<40} "
                f"{voice.gender or 'N/A':<15} {voice.language or 'N/A'}"
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
