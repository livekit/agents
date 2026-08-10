#!/usr/bin/env python3
"""
Direct E2E test for Speechify streaming endpoint migration.
Tests the endpoint directly without full agents framework.
"""

import asyncio
import base64
import json
import os
import sys

import httpx


async def test_streaming_endpoint():
    """Test the new /v1/audio/stream/with-timestamps endpoint."""
    print("=== Testing /v1/audio/stream/with-timestamps ===")

    api_key = os.environ.get("SPEECHIFY_API_KEY")
    if not api_key:
        print("ERROR: SPEECHIFY_API_KEY not set")
        sys.exit(1)

    url = "https://api.sws.speechify.com/v1/audio/stream/with-timestamps"

    request_body = {
        "input": "Hello world! This is a test.",
        "voice_id": "dominic_32",
        "model": "simba-3.2",
        "output_format": "pcm_24000",
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Speechify-Caller": "livekit",
    }

    audio_chunks = []
    speech_marks = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", url, json=request_body, headers=headers) as response:
            print(f"Response status: {response.status_code}")
            if response.status_code != 200:
                error = await response.aread()
                print(f"Error: {error.decode('utf-8')}")
                sys.exit(1)

            event_type = None
            async for line in response.aiter_lines():
                line = line.strip()

                if not line:
                    continue

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()

                    try:
                        parsed = json.loads(data_str)

                        if event_type == "speech.chunk":
                            if "audio" in parsed:
                                audio_b64 = parsed["audio"]
                                audio_bytes = base64.b64decode(audio_b64)
                                audio_chunks.append(audio_bytes)
                                print(
                                    f"  Audio chunk: {len(audio_bytes)} bytes (decoded from base64)"
                                )

                            if "speech_marks" in parsed:
                                marks = parsed["speech_marks"]
                                speech_marks.extend(marks)
                                for mark in marks:
                                    if mark.get("type") == "word":
                                        print(
                                            f"  Word: '{mark.get('value')}' @ {mark.get('start')}ms"
                                        )
                    except json.JSONDecodeError:
                        pass

    total_audio = sum(len(chunk) for chunk in audio_chunks)
    print(f"\n✓ Total audio: {total_audio} bytes")
    print(f"✓ Speech marks: {len(speech_marks)} marks")

    assert total_audio > 0, "No audio generated"
    assert len(speech_marks) > 0, "No speech marks"

    print("\n✓ Streaming endpoint test passed!")


async def main():
    try:
        await test_streaming_endpoint()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
