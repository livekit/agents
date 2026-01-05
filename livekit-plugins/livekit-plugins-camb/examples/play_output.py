#!/usr/bin/env python3
"""
Play the generated output.wav file using system audio player.

Usage:
    cd livekit-plugins/livekit-plugins-camb
    uv run python examples/play_output.py
"""

import os
import platform
import subprocess
import sys


def main():
    wav_file = "examples/output.wav"

    if not os.path.exists(wav_file):
        print(f"Error: {wav_file} not found. Run simple_demo.py first.")
        sys.exit(1)

    system = platform.system()
    print(f"Playing {wav_file}...")

    if system == "Darwin":  # macOS
        subprocess.run(["afplay", wav_file])
    elif system == "Linux":
        # Try multiple players
        for player in ["aplay", "paplay", "ffplay", "mpv"]:
            try:
                subprocess.run([player, wav_file], check=True)
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    elif system == "Windows":
        import winsound

        winsound.PlaySound(wav_file, winsound.SND_FILENAME)
    else:
        print(f"Unsupported system: {system}")
        sys.exit(1)

    print("✅ Playback complete!")


if __name__ == "__main__":
    main()
