#!/usr/bin/env python3
"""Test script for Krisp VIVA turn detection with real audio files.

This script processes audio files through Krisp VIVA turn detector and analyzes
turn detection, allowing you to test turn detection on real audio data.

Usage:
    python test_viva_turn_audiofile.py input.wav
    python test_viva_turn_audiofile.py input.wav --threshold 0.7
    python test_viva_turn_audiofile.py input.wav --frame-duration 20

Requirements:
    pip install soundfile numpy livekit-plugins-krisp
    Set KRISP_VIVA_SDK_LICENSE_KEY environment variable with your Krisp license key
    Set KRISP_VIVA_TURN_MODEL_PATH environment variable to point to your .kef model file

Note: This is a standalone script, not a pytest test. It will be skipped during pytest collection.
"""

import argparse
import os
import sys
import time

# This is a standalone script, not a pytest test file
# Handle missing dependencies gracefully during pytest collection
try:
    import numpy as np

    try:
        import soundfile as sf  # noqa: F401
    except ImportError:
        sf = None  # Will be checked later if needed
except ImportError as e:
    # Check if we're being imported by pytest
    if "pytest" in sys.modules or "_pytest" in sys.modules:
        import pytest

        pytest.skip(
            f"Dependencies not available: {e}. "
            "This is a standalone script (not a pytest test) and requires soundfile.",
            allow_module_level=True,
        )
    else:
        # If running as standalone script, raise the error immediately
        raise

from livekit import rtc

# Import audio utilities - handle both standalone and module usage
try:
    from .audio_file_utils import read_audio_file
except ImportError:
    # Running as standalone script
    from audio_file_utils import read_audio_file


def load_turn_detector():
    """Load the Krisp VIVA turn detector module."""
    # Check license key
    license_key = os.getenv("KRISP_VIVA_SDK_LICENSE_KEY")
    if not license_key:
        print("Warning: KRISP_VIVA_SDK_LICENSE_KEY environment variable not set")
        print("Set it with: export KRISP_VIVA_SDK_LICENSE_KEY=your-license-key")
        print("Or in PowerShell: $env:KRISP_VIVA_SDK_LICENSE_KEY='your-license-key'")
        print("The SDK may fail to initialize without a valid license key.")
    
    env_var = "KRISP_VIVA_TURN_MODEL_PATH"

    # Check model path
    model_path = os.getenv(env_var)
    if not model_path:
        print(f"Error: {env_var} environment variable not set")
        print(f"Set it with: export {env_var}=/path/to/model.kef")
        print(f"Or in PowerShell: $env:{env_var}='C:\\path\\to\\model.kef'")
        sys.exit(1)

    # Check if model file exists
    if not os.path.isfile(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # Import the module
    try:
        from livekit.plugins.krisp import KrispVivaTurn
    except ImportError as e:
        print(f"Error: Could not import KrispVivaTurn: {e}")
        print("Make sure livekit-plugins-krisp is installed: pip install livekit-plugins-krisp")
        sys.exit(1)

    return KrispVivaTurn, model_path


def analyze_audio_file(
    input_path: str,
    threshold: float = 0.5,
    frame_duration_ms: int = 20,
    chunk_duration_ms: int = 20,
    verbose: bool = False,
    output_file: str | None = None,
) -> None:
    """Analyze an audio file for turn detection using Krisp VIVA turn detector.

    Args:
        input_path: Path to input audio file
        threshold: Probability threshold for turn completion (0.0 to 1.0)
        frame_duration_ms: Frame duration in milliseconds for turn detection model
        chunk_duration_ms: Processing chunk size in milliseconds
        verbose: Show detailed processing information
        output_file: Optional path to save turn probabilities (one per line)
    """
    # Read audio file using shared utility
    audio_data, sample_rate = read_audio_file(input_path, verbose=True)

    # Load turn detector
    DetectorClass, model_path = load_turn_detector()

    # Check if sample rate is supported
    supported_rates = [8000, 16000, 24000, 32000, 44100, 48000]
    if sample_rate not in supported_rates:
        print(f"Warning: Sample rate {sample_rate} not in supported rates {supported_rates}")
        print("Resampling may be required. Continuing anyway...")

    print("\nInitializing VIVA turn detector:")
    print(f"  - Model path: {model_path}")
    print(f"  - Threshold: {threshold}")
    print(f"  - Frame duration: {frame_duration_ms}ms")
    print(f"  - Sample rate: {sample_rate}Hz")
    print(f"  - Processing chunk size: {chunk_duration_ms}ms")

    # Create turn detector instance
    print("\nInitializing turn detector...")
    init_start_time = time.time()
    turn_detector = DetectorClass(
        model_path=model_path,
        threshold=threshold,
        frame_duration_ms=frame_duration_ms,
        sample_rate=sample_rate,
    )
    init_duration = time.time() - init_start_time
    print(f"Turn detector initialized in {init_duration * 1000:.2f}ms")

    try:
        print("\nDetector info:")
        print(f"  - Model: {turn_detector.model}")
        print(f"  - Provider: {turn_detector.provider}")
        print(f"  - Threshold: {turn_detector.threshold}")

        print("\nProcessing audio for turn detection...")

        # Calculate exact frame size based on frame duration
        frame_size_samples = int(sample_rate * frame_duration_ms / 1000)
        print(f"  Frame size: {frame_size_samples} samples ({frame_duration_ms}ms)")

        turn_events = []
        speech_segments = []
        current_speech_start = None
        all_probabilities = []  # Store all probabilities for output file

        # Simple energy-based VAD (for demonstration)
        energy_threshold = np.std(audio_data) * 0.1

        # Buffer for incomplete frames
        audio_buffer = np.array([], dtype=np.int16)
        frames_processed = 0

        # Process audio in chunks
        read_chunk_size = max(frame_size_samples, int(sample_rate * chunk_duration_ms / 1000))

        for i in range(0, len(audio_data), read_chunk_size):
            chunk = audio_data[i : i + read_chunk_size]

            if len(chunk) == 0:
                break

            # Add chunk to buffer
            audio_buffer = np.concatenate([audio_buffer, chunk])

            # Process complete frames from buffer
            while len(audio_buffer) >= frame_size_samples:
                # Extract exactly one frame
                frame_samples = audio_buffer[:frame_size_samples].copy()
                audio_buffer = audio_buffer[frame_size_samples:]

                # Calculate timestamp for this frame
                timestamp = frames_processed * frame_duration_ms / 1000.0
                frames_processed += 1

                # Simple VAD: check if frame has significant energy
                frame_energy = np.sqrt(np.mean(frame_samples.astype(np.float32) ** 2))
                is_speech = frame_energy > energy_threshold

                # Create LiveKit AudioFrame
                audio_frame = rtc.AudioFrame(
                    data=frame_samples.tobytes(),
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=frame_size_samples,
                )

                # Process frame through turn detector
                probability = turn_detector.process_audio(audio_frame, is_speech=is_speech)

                # Collect probabilities (skip negative values during warmup)
                if probability >= 0:
                    all_probabilities.extend(turn_detector.frame_probabilities)

                # Track speech segments
                if is_speech:
                    if current_speech_start is None:
                        current_speech_start = timestamp
                else:
                    if current_speech_start is not None:
                        speech_segments.append((current_speech_start, timestamp))
                        current_speech_start = None

                # Track turn completion events
                # Check both the returned probability and speech_triggered flag
                if turn_detector.speech_triggered and probability >= threshold:
                    turn_events.append(
                        {
                            "timestamp": timestamp,
                            "probability": probability,
                            "speech_triggered": turn_detector.speech_triggered,
                        }
                    )
                    if verbose:
                        print(f"  Turn completed at {timestamp:.2f}s (prob={probability:.3f})")
                    # Clear state after turn completion (like Pipecat does)
                    turn_detector.clear()

            # Progress indicator
            if i % (read_chunk_size * 50) == 0:
                progress = (i / len(audio_data)) * 100
                print(f"  Progress: {progress:.1f}%", end="\r")

        # Process any remaining incomplete frame
        if len(audio_buffer) > 0:
            if verbose:
                print(
                    f"\n  Warning: {len(audio_buffer)} samples remaining "
                    f"(incomplete frame, will be discarded)"
                )

        print("  Progress: 100.0%")

        # Final speech segment if still speaking
        if current_speech_start is not None:
            speech_segments.append((current_speech_start, len(audio_data) / sample_rate))

        # Print results
        print("\n" + "=" * 60)
        print("Turn Detection Results:")
        print("=" * 60)

        print(f"\nSpeech Segments Detected: {len(speech_segments)}")
        for i, (start, end) in enumerate(speech_segments, 1):
            duration = end - start
            print(f"  Segment {i}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")

        print(f"\nTurn Completion Events: {len(turn_events)}")
        for i, event in enumerate(turn_events, 1):
            print(
                f"  Turn {i} completed at {event['timestamp']:.2f}s "
                f"(probability: {event['probability']:.3f})"
            )

        print("\nFinal State:")
        print(f"  Speech triggered: {turn_detector.speech_triggered}")
        print(f"  Last probability: {turn_detector.last_probability}")
        print(f"  Total probabilities collected: {len(all_probabilities)}")

        if len(turn_events) == 0:
            print("\n  ⚠️  No turn completion events detected.")
            print("     This could mean:")
            print("     - The audio doesn't contain clear turn boundaries")
            print("     - The threshold is too high")
            print("     - The model needs different parameters")

        # Save probabilities to output file if specified
        if output_file:
            with open(output_file, "w") as f:
                for prob in all_probabilities:
                    f.write(f"{prob}\n")
            print(f"\n📄 Turn probabilities saved to: {output_file}")
            print(f"   Total frames: {len(all_probabilities)}")

        print("\n✅ Analysis complete!")

    finally:
        # Cleanup
        turn_detector.close()
        print("Turn detector closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Test Krisp VIVA turn detector with real audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_viva_turn_audiofile.py conversation.wav
  python test_viva_turn_audiofile.py input.wav --threshold 0.7
  python test_viva_turn_audiofile.py input.wav --frame-duration 20

Supported audio formats: WAV, FLAC, OGG, etc. (via soundfile)
Supported sample rates: 8000, 16000, 24000, 32000, 44100, 48000 Hz

Note: 
    - Set KRISP_VIVA_SDK_LICENSE_KEY environment variable with your Krisp license key
    - Set KRISP_VIVA_TURN_MODEL_PATH environment variable to point to your .kef model file
        """,
    )

    parser.add_argument("input", help="Input audio file path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for turn completion (0.0 to 1.0, default: 0.5)",
    )
    parser.add_argument(
        "--frame-duration",
        type=int,
        default=20,
        choices=[10, 15, 20, 30, 32],
        help="Frame duration in milliseconds for turn detection model (default: 20)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=20,
        help="Processing chunk size in milliseconds (default: 20)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed processing information",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path to save turn probabilities (one probability per line)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print(f"Error: Threshold must be between 0.0 and 1.0, got {args.threshold}")
        sys.exit(1)

    # Process the audio
    analyze_audio_file(
        args.input,
        threshold=args.threshold,
        frame_duration_ms=args.frame_duration,
        chunk_duration_ms=args.chunk_duration,
        verbose=args.verbose,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
