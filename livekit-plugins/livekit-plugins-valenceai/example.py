"""
Example usage of the ValenceAI plugin for LiveKit Agents.

This example demonstrates how to use the ValenceAI plugin to detect emotions
in audio using the LiveKit Agents framework.
"""

import asyncio

from livekit.plugins.valenceai import STT


async def main():
    # Initialize the ValenceAI STT with your API key
    stt = STT(
        api_key="your-api-key-here",  # or set VALENCE_API_KEY env var
        show_progress=True,
        sample_rate=16000,
    )

    # Example: Process an audio file
    # In a real scenario, you would get the audio from LiveKit room
    # For this example, we'll show how to use it with an AudioBuffer

    print("ValenceAI STT initialized")
    print(f"Provider: {stt.provider}")
    print(f"Model: {stt.model}")

    # Note: In a real LiveKit agent, you would:
    # 1. Receive audio from participants
    # 2. Buffer the audio
    # 3. Call stt.recognize() on the buffer
    # 4. Extract emotion data from the result

    # Example of processing audio buffer (pseudo-code)
    # audio_buffer = AudioBuffer()
    # ... collect audio frames ...
    # result = await stt.recognize(audio_buffer)
    # emotions = result.metadata
    # print(f"Detected emotions: {emotions}")


if __name__ == "__main__":
    asyncio.run(main())
