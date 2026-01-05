#!/usr/bin/env python3
"""
Quick test script for Camb.ai TTS plugin.

Usage:
    export CAMB_API_KEY=your_key_here
    export PATH="$HOME/.local/bin:$PATH"
    cd livekit-plugins/livekit-plugins-camb
    uv run python examples/test_camb_tts.py
"""

import asyncio
import os
import sys
import wave

# Load .env file from examples directory
from pathlib import Path

try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not required, can use export instead

# Add parent directory to path to import the plugin
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from livekit.plugins.camb import TTS, list_voices


async def test_list_voices():
    """Test voice listing."""
    print("=== Testing list_voices() ===\n")

    try:
        voices = await list_voices()
        print(f"✅ Found {len(voices)} voices\n")

        # Show first 5 voices
        for i, voice in enumerate(voices[:5], 1):
            print(
                f"{i}. {voice.name:30s} | ID: {voice.id:8d} | "
                f"{voice.gender or 'N/A':10s} | {voice.language or 'N/A'}"
            )

        if len(voices) > 5:
            print(f"\n... and {len(voices) - 5} more voices")

        return voices[0].id if voices else None

    except Exception as e:
        print(f"❌ Error listing voices: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_basic_synthesis(voice_id: int | None = None):
    """Test basic text synthesis."""
    print("\n=== Testing Basic Synthesis ===\n")

    try:
        # Use voice ID 4277 (ack-noisy) for testing
        if voice_id is None:
            voice_id = 4277

        tts = TTS(voice_id=voice_id)
        print(f"Using voice ID: {voice_id}")

        # Synthesize
        text = "Hello from Camb.ai! This is a test of the text to speech system."
        print(f'\nSynthesizing: "{text}"\n')

        stream = tts.synthesize(text)
        audio = await stream.collect()

        print("✅ Generated audio:")
        print(f"   Duration: {audio.duration:.2f}s")
        print(f"   Data size: {len(audio.data)} bytes")
        print(f"   Sample rate: {audio.sample_rate} Hz")
        print(f"   Channels: {audio.num_channels}")

        # Save to file with proper WAV headers
        output_file = "examples/test_output.wav"
        with wave.open(output_file, "wb") as wav_file:
            wav_file.setnchannels(audio.num_channels)
            wav_file.setsampwidth(2)  # 16-bit PCM = 2 bytes
            wav_file.setframerate(audio.sample_rate)
            wav_file.writeframes(audio.data)
        print(f"\n   Saved to: {output_file}")

        # Play the audio
        import platform
        import subprocess

        print("\n   Playing audio...")
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["afplay", output_file])
        print("   ✅ Playback complete!")

    except Exception as e:
        print(f"❌ Error during synthesis: {e}")
        import traceback

        traceback.print_exc()


async def test_mars_models():
    """Test different MARS models."""
    print("\n=== Testing MARS Models ===\n")

    models = [
        ("mars-8", "Default balanced model"),
        ("mars-8-flash", "Faster inference"),
        ("mars-8-instruct", "With user instructions"),
    ]

    import platform
    import subprocess

    for model, desc in models:
        try:
            print(f"\nTesting {model} ({desc})...")

            if model == "mars-8-instruct":
                tts = TTS(
                    voice_id=4277,
                    model=model,
                    user_instructions="Speak enthusiastically and clearly",
                )
            else:
                tts = TTS(voice_id=4277, model=model)

            stream = tts.synthesize("Testing the MARS model.")
            audio = await stream.collect()

            # Save to file with proper WAV headers
            output_file = f"examples/{model}_output.wav"
            with wave.open(output_file, "wb") as wav_file:
                wav_file.setnchannels(audio.num_channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(audio.sample_rate)
                wav_file.writeframes(audio.data)

            print(f"✅ {model}: {audio.duration:.2f}s, {len(audio.data)} bytes")
            print(f"   Saved to: {output_file}")

            # Play the audio
            print(f"   Playing {model}...")
            if platform.system() == "Darwin":
                subprocess.run(["afplay", output_file])
            print("   ✅ Playback complete!")

        except Exception as e:
            print(f"❌ {model} failed: {e}")


async def test_speed_control():
    """Test speech speed control."""
    print("\n=== Testing Speed Control ===\n")

    speeds = [0.75, 1.0, 1.25, 1.5]
    text = "The quick brown fox jumps over the lazy dog."

    import platform
    import subprocess

    for speed in speeds:
        try:
            tts = TTS(voice_id=4277, speed=speed)
            stream = tts.synthesize(text)
            audio = await stream.collect()

            # Save to file with proper WAV headers
            output_file = f"examples/speed_{speed:.2f}_output.wav"
            with wave.open(output_file, "wb") as wav_file:
                wav_file.setnchannels(audio.num_channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(audio.sample_rate)
                wav_file.writeframes(audio.data)

            print(f"✅ Speed {speed:0.2f}: {audio.duration:.2f}s")
            print(f"   Saved to: {output_file}")

            # Play the audio
            print(f"   Playing speed {speed}...")
            if platform.system() == "Darwin":
                subprocess.run(["afplay", output_file])
            print("   ✅ Playback complete!")

        except Exception as e:
            print(f"❌ Speed {speed} failed: {e}")


async def test_languages():
    """Test multi-language support."""
    print("\n=== Testing Multi-Language ===\n")

    tests = [
        ("en-us", "Hello, how are you?"),
        ("fr-fr", "Bonjour, comment allez-vous?"),
        ("es-es", "Hola, ¿cómo estás?"),
    ]

    import platform
    import subprocess

    for lang, text in tests:
        try:
            tts = TTS(voice_id=4277, language=lang)
            stream = tts.synthesize(text)
            audio = await stream.collect()

            # Save to file with proper WAV headers
            output_file = f"examples/{lang}_output.wav"
            with wave.open(output_file, "wb") as wav_file:
                wav_file.setnchannels(audio.num_channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(audio.sample_rate)
                wav_file.writeframes(audio.data)

            print(f"✅ {lang}: {audio.duration:.2f}s")
            print(f"   Saved to: {output_file}")

            # Play the audio
            print(f"   Playing {lang}...")
            if platform.system() == "Darwin":
                subprocess.run(["afplay", output_file])
            print("   ✅ Playback complete!")

        except Exception as e:
            print(f"❌ {lang} failed: {e}")


async def main():
    """Run all tests."""
    # Check API key
    if not os.getenv("CAMB_API_KEY"):
        print("❌ CAMB_API_KEY environment variable not set!")
        print("\nUsage:")
        print("  export CAMB_API_KEY=your_key_here")
        print('  export PATH="$HOME/.local/bin:$PATH"')
        print("  cd livekit-plugins/livekit-plugins-camb")
        print("  uv run python examples/test_camb_tts.py")
        return

    print("🚀 Camb.ai TTS Plugin Test Suite\n")
    print("=" * 60)

    # Test 1: List voices
    voice_id = await test_list_voices()

    # Test 2: Basic synthesis
    await test_basic_synthesis(voice_id)

    # Test 3: MARS models
    await test_mars_models()

    # Test 4: Speed control
    await test_speed_control()

    # Test 5: Multi-language (optional - uncomment to test)
    # await test_languages()

    print("\n" + "=" * 60)
    print("✅ Test suite complete!")


if __name__ == "__main__":
    asyncio.run(main())
