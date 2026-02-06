#!/usr/bin/env python3
"""Test script for Krisp VIVA noise reduction with real audio files.

This script processes audio files through Krisp VIVA filter and outputs
the noise-reduced audio, allowing you to test noise reduction on real audio data.

Usage:
    python test_viva_filter_audiofile.py input.wav output.wav
    python test_viva_filter_audiofile.py input.wav output.wav --level 80
    python test_viva_filter_audiofile.py input.wav output.wav --frame-duration 20

Requirements:
    pip install soundfile numpy livekit-plugins-krisp
    Set KRISP_VIVA_SDK_LICENSE_KEY environment variable with your Krisp license key
    Set KRISP_VIVA_FILTER_MODEL_PATH environment variable to point to your .kef model file

Note: This is a standalone script, not a pytest test. It will be skipped during pytest collection.
"""

import argparse
import asyncio
import os
import sys
import time

# Add package root to Python path when running as standalone script
# This allows imports like 'livekit.plugins.krisp' to work
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up from tests/ -> krisp/ -> plugins/ -> livekit/ -> livekit-plugins-krisp/
_package_root = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", ".."))
if _package_root not in sys.path:
    sys.path.insert(0, _package_root)

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

from livekit import rtc  # noqa: E402

# Import audio utilities - handle both standalone and module usage
try:
    from .audio_file_utils import calculate_audio_stats, read_audio_file, write_audio_file
except ImportError:
    # Running as standalone script
    from audio_file_utils import calculate_audio_stats, read_audio_file, write_audio_file


def load_filter():
    """Load the Krisp VIVA filter module."""
    # Check license key
    license_key = os.getenv("KRISP_VIVA_SDK_LICENSE_KEY")
    if not license_key:
        print("Warning: KRISP_VIVA_SDK_LICENSE_KEY environment variable not set")
        print("Set it with: export KRISP_VIVA_SDK_LICENSE_KEY=your-license-key")
        print("Or in PowerShell: $env:KRISP_VIVA_SDK_LICENSE_KEY='your-license-key'")
        print("The SDK may fail to initialize without a valid license key.")
    
    env_var = "KRISP_VIVA_FILTER_MODEL_PATH"

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
        from livekit.plugins.krisp import KrispVivaFilterFrameProcessor
    except ImportError as e:
        print(f"Error: Could not import KrispVivaFilterFrameProcessor: {e}")
        print("Make sure livekit-plugins-krisp is installed: pip install livekit-plugins-krisp")
        sys.exit(1)

    return KrispVivaFilterFrameProcessor, model_path


async def process_audio_file(
    input_path: str,
    output_path: str,
    noise_level: int = 100,
    frame_duration_ms: int = 20,
    chunk_duration_ms: int = 20,
    verbose: bool = False,
) -> None:
    """Process an audio file through Krisp VIVA noise reduction filter.

    Args:
        input_path: Path to input audio file
        output_path: Path to output (filtered) audio file
        noise_level: Noise suppression level (0-100, higher = more suppression)
        frame_duration_ms: Frame duration in milliseconds for filter
        chunk_duration_ms: Processing chunk size in milliseconds
        verbose: Show detailed processing information
    """
    # Read audio file using shared utility
    audio_data, sample_rate = read_audio_file(input_path, verbose=True)

    # Load filter
    FilterClass, model_path = load_filter()

    # Check if sample rate is supported
    supported_rates = [8000, 16000, 24000, 32000, 44100, 48000]
    if sample_rate not in supported_rates:
        print(f"Warning: Sample rate {sample_rate} not in supported rates {supported_rates}")
        print("Resampling may be required. Continuing anyway...")

    print("\nInitializing VIVA filter:")
    print(f"  - Model path: {model_path}")
    print(f"  - Noise suppression level: {noise_level}")
    print(f"  - Frame duration: {frame_duration_ms}ms")
    print(f"  - Sample rate: {sample_rate}Hz")
    print(f"  - Processing chunk size: {chunk_duration_ms}ms")

    # Create filter instance
    print("\nInitializing filter...")
    init_start_time = time.time()
    audio_filter = FilterClass(
        model_path=model_path,
        noise_suppression_level=noise_level,
        frame_duration_ms=frame_duration_ms,
        sample_rate=sample_rate,
    )
    init_duration = time.time() - init_start_time
    print(f"Filter initialized in {init_duration * 1000:.2f}ms")

    try:
        print("\nFilter info:")
        print(f"  - Status: {('Enabled' if audio_filter.is_enabled else 'Disabled')}")

        print("\nProcessing audio through noise reduction filter...")

        # Calculate exact frame size based on frame duration
        frame_size_samples = int(sample_rate * frame_duration_ms / 1000)
        print(f"  Frame size: {frame_size_samples} samples ({frame_duration_ms}ms)")

        # Output buffer
        filtered_audio = []

        # Buffer for incomplete frames
        audio_buffer = np.array([], dtype=np.int16)
        frames_processed = 0

        # Process audio in chunks
        read_chunk_size = max(frame_size_samples, int(sample_rate * chunk_duration_ms / 1000))

        process_start_time = time.time()

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

                frames_processed += 1

                # Create LiveKit AudioFrame
                audio_frame = rtc.AudioFrame(
                    data=frame_samples.tobytes(),
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=frame_size_samples,
                )

                # Process the frame (FrameProcessor.process() is synchronous)
                filtered_frame = audio_filter.process(audio_frame)

                # Convert filtered frame back to numpy array
                filtered_samples = np.frombuffer(filtered_frame.data, dtype=np.int16)
                filtered_audio.append(filtered_samples)

            # Progress indicator
            if i % (read_chunk_size * 50) == 0:
                progress = (i / len(audio_data)) * 100
                print(f"  Progress: {progress:.1f}%", end="\r")

        # Add any remaining incomplete frame (unfiltered)
        if len(audio_buffer) > 0:
            if verbose:
                print(
                    f"\n  Warning: {len(audio_buffer)} samples remaining "
                    f"(incomplete frame, added unfiltered)"
                )
            filtered_audio.append(audio_buffer)

        print("  Progress: 100.0%")

        process_duration = time.time() - process_start_time

        # Concatenate all filtered frames
        filtered_audio_data = np.concatenate(filtered_audio)

        # Calculate audio statistics
        input_stats = calculate_audio_stats(audio_data)
        output_stats = calculate_audio_stats(filtered_audio_data)

        print("\n" + "=" * 60)
        print("Processing Results:")
        print("=" * 60)
        print("\nInput:")
        print(f"  - Samples: {input_stats['samples']}")
        print(f"  - Duration: {input_stats['samples'] / sample_rate:.2f}s")
        print(f"  - RMS level: {input_stats['rms']:.2f}")
        print(f"  - Range: {input_stats['min']} to {input_stats['max']}")

        print("\nOutput:")
        print(f"  - Samples: {output_stats['samples']}")
        print(f"  - Duration: {output_stats['samples'] / sample_rate:.2f}s")
        print(f"  - RMS level: {output_stats['rms']:.2f}")
        print(f"  - Range: {output_stats['min']} to {output_stats['max']}")

        print("\nPerformance:")
        print(f"  - Frames processed: {frames_processed}")
        print(f"  - Processing time: {process_duration:.2f}s")
        print(f"  - Real-time factor: {(len(audio_data) / sample_rate) / process_duration:.2f}x")

        # Save filtered audio using shared utility
        write_audio_file(output_path, filtered_audio_data, sample_rate, verbose=True)

        print("\n✅ Processing complete!")

    finally:
        # Cleanup
        audio_filter.close()
        print("Filter closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Test Krisp VIVA noise reduction with real audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_viva_filter_audiofile.py input.wav output.wav
  python test_viva_filter_audiofile.py input.wav output.wav --level 80
  python test_viva_filter_audiofile.py input.wav output.wav --frame-duration 20

Supported audio formats: WAV, FLAC, OGG, etc. (via soundfile)
Supported sample rates: 8000, 16000, 24000, 32000, 44100, 48000 Hz

Note: 
    - Set KRISP_VIVA_SDK_LICENSE_KEY environment variable with your Krisp license key
    - Set KRISP_VIVA_FILTER_MODEL_PATH environment variable to point to your .kef model file
        """,
    )

    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("output", help="Output (filtered) audio file path")
    parser.add_argument(
        "--level",
        type=int,
        default=100,
        help="Noise suppression level (0-100, higher = more suppression, default: 100)",
    )
    parser.add_argument(
        "--frame-duration",
        type=int,
        default=20,
        choices=[10, 15, 20, 30, 32],
        help="Frame duration in milliseconds for filter (default: 20)",
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

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Validate noise level
    if not 0 <= args.level <= 100:
        print(f"Error: Noise level must be between 0 and 100, got {args.level}")
        sys.exit(1)

    # Check if output file already exists
    if os.path.exists(args.output):
        response = input(f"Warning: Output file '{args.output}' already exists. Overwrite? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Process the audio
    asyncio.run(
        process_audio_file(
            args.input,
            args.output,
            noise_level=args.level,
            frame_duration_ms=args.frame_duration,
            chunk_duration_ms=args.chunk_duration,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
