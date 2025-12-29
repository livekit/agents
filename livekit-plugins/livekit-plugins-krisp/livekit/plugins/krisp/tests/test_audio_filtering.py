#!/usr/bin/env python3
"""Test script for Krisp VIVA filter with real audio data.

This script processes audio files through the Krisp filter and saves the output,
allowing you to compare the original and filtered audio.

Usage:
    python test_audio_filtering.py input.wav output.wav

Requirements:
    pip install livekit-plugins-krisp soundfile matplotlib scipy
"""

import argparse
import asyncio
import os
import sys

import numpy as np
import soundfile as sf

from livekit import rtc

try:
    from livekit.plugins.krisp import KrispVivaFilter
except ImportError:
    print("Error: livekit-plugins-krisp not installed")
    print("Install with: pip install -e .")
    sys.exit(1)


async def process_audio_file(
    input_path: str,
    output_path: str,
    noise_suppression_level: int = 100,
    frame_duration_ms: int = 20,
    show_visualization: bool = False,
    verbose: bool = False,
) -> None:
    """Process an audio file through the Krisp filter.

    Args:
        input_path: Path to input audio file
        output_path: Path to save filtered audio
        noise_suppression_level: Noise suppression level (0-100)
        frame_duration_ms: Frame duration in milliseconds
        show_visualization: Whether to show before/after spectrograms
        verbose: Show detailed processing information
    """
    print(f"Loading audio from: {input_path}")

    # Read the audio file
    audio_data, sample_rate = sf.read(input_path, dtype="int16")

    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        print(f"Converting from {audio_data.shape[1]} channels to mono")
        audio_data = audio_data.mean(axis=1).astype(np.int16)

    print(
        f"Audio info: {len(audio_data)} samples, {sample_rate} Hz, {len(audio_data) / sample_rate:.2f} seconds"
    )

    # Check if sample rate is supported
    supported_rates = [8000, 16000, 24000, 32000, 44100, 48000]
    if sample_rate not in supported_rates:
        print(f"Warning: Sample rate {sample_rate} not in supported rates {supported_rates}")
        print("Resampling may be required. Continuing anyway...")

    # Initialize Krisp filter (with sample rate for immediate session creation)
    print("\nInitializing Krisp filter:")
    print(f"  - Noise suppression level: {noise_suppression_level}")
    print(f"  - Frame duration: {frame_duration_ms}ms")
    print(f"  - Sample rate: {sample_rate}Hz")

    krisp_filter = KrispVivaFilter(
        noise_suppression_level=noise_suppression_level,
        frame_duration_ms=frame_duration_ms,
        sample_rate=sample_rate,  # Create session immediately
    )

    try:
        # Process audio in chunks
        print("\nProcessing audio...")
        filtered_samples = []
        total_frames = 0
        empty_frames = 0

        # Use chunk size matching Krisp frame duration for optimal processing
        chunk_size = int(sample_rate * frame_duration_ms / 1000)
        print(f"  - Chunk size: {chunk_size} samples ({frame_duration_ms}ms)")

        if verbose:
            print(f"  - Processing {len(audio_data)} samples in chunks of {chunk_size}")

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]

            if len(chunk) == 0:
                break

            # Skip incomplete chunks (last chunk might be smaller)
            if len(chunk) < chunk_size:
                if verbose:
                    print(f"\n  Skipping incomplete final chunk: {len(chunk)} samples")
                break

            # Create AudioFrame
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(chunk),
            )

            # Filter the frame
            filtered_frame = await krisp_filter.filter(frame)

            # Collect filtered samples
            if filtered_frame.samples_per_channel > 0:
                filtered_chunk = np.frombuffer(filtered_frame.data, dtype=np.int16)
                filtered_samples.append(filtered_chunk)
                total_frames += 1

                if verbose and total_frames <= 3:
                    print(
                        f"    Frame {total_frames}: {len(chunk)} -> {len(filtered_chunk)} samples"
                    )
            else:
                empty_frames += 1

            # Progress indicator
            if i % (chunk_size * 50) == 0:  # Every second at 20ms chunks
                progress = (i / len(audio_data)) * 100
                print(f"  Progress: {progress:.1f}%", end="\r")

        print(f"  Progress: 100.0% - Processed {total_frames} frames")
        if empty_frames > 0:
            print(f"  Note: {empty_frames} frames were skipped (frame size mismatch)")

        # Concatenate all filtered samples
        if filtered_samples:
            filtered_audio = np.concatenate(filtered_samples)
            print(f"\nFiltered audio: {len(filtered_audio)} samples")

            # Save the filtered audio
            print(f"Saving filtered audio to: {output_path}")
            sf.write(output_path, filtered_audio, sample_rate)

            # Calculate statistics
            original_rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            filtered_rms = np.sqrt(np.mean(filtered_audio.astype(np.float32) ** 2))

            print("\nAudio Statistics:")
            print(f"  Original RMS: {original_rms:.2f}")
            print(f"  Filtered RMS: {filtered_rms:.2f}")
            print(f"  RMS Ratio: {filtered_rms / original_rms:.2f}")

            if filtered_rms < 0.01:
                print("\n  ⚠️  WARNING: Filtered audio is very quiet or silent!")
                print("     This may indicate a processing issue.")

            print("\n✅ Processing complete!")
            print(f"   Original: {input_path}")
            print(f"   Filtered: {output_path}")
            print("\nListen to both files to compare the noise reduction.")

            # Close Krisp filter BEFORE showing visualization
            # This prevents segfault when matplotlib window is closed
            krisp_filter.close()
            print("Krisp filter closed.")

            # Show visualization if requested (after Krisp cleanup)
            if show_visualization:
                visualize_audio_comparison(audio_data, filtered_audio, sample_rate)

        else:
            print("Error: No filtered audio produced")
            sys.exit(1)

    finally:
        # Ensure cleanup even if there was an error
        try:
            krisp_filter.close()
        except Exception:
            pass  # Already closed or error during close


def visualize_audio_comparison(
    original: np.ndarray, filtered: np.ndarray, sample_rate: int
) -> None:
    """Create before/after spectrograms for visual comparison.

    Args:
        original: Original audio samples
        filtered: Filtered audio samples
        sample_rate: Sample rate in Hz
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import signal
    except ImportError:
        print("Warning: matplotlib or scipy not installed. Skipping visualization.")
        print("Install with: pip install matplotlib scipy")
        return

    print("\nGenerating spectrograms...")

    # Ensure same length for comparison
    min_len = min(len(original), len(filtered))
    original = original[:min_len]
    filtered = filtered[:min_len]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot waveforms
    time = np.arange(len(original)) / sample_rate

    axes[0, 0].plot(time, original, alpha=0.7)
    axes[0, 0].set_title("Original Waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time, filtered, alpha=0.7, color="green")
    axes[0, 1].set_title("Filtered Waveform")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot spectrograms
    nperseg = 2048
    f_orig, t_orig, Sxx_orig = signal.spectrogram(
        original.astype(np.float32), sample_rate, nperseg=nperseg
    )
    f_filt, t_filt, Sxx_filt = signal.spectrogram(
        filtered.astype(np.float32), sample_rate, nperseg=nperseg
    )

    im1 = axes[1, 0].pcolormesh(
        t_orig, f_orig, 10 * np.log10(Sxx_orig + 1e-10), shading="gouraud", cmap="viridis"
    )
    axes[1, 0].set_title("Original Spectrogram")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Frequency (Hz)")
    axes[1, 0].set_ylim([0, 8000])  # Focus on speech frequencies
    plt.colorbar(im1, ax=axes[1, 0], label="Power (dB)")

    im2 = axes[1, 1].pcolormesh(
        t_filt, f_filt, 10 * np.log10(Sxx_filt + 1e-10), shading="gouraud", cmap="viridis"
    )
    axes[1, 1].set_title("Filtered Spectrogram")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Frequency (Hz)")
    axes[1, 1].set_ylim([0, 8000])
    plt.colorbar(im2, ax=axes[1, 1], label="Power (dB)")

    plt.tight_layout()

    # Save the figure
    output_path = "krisp_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Spectrogram comparison saved to: {output_path}")

    # Try to show the plot
    try:
        plt.show()
    except Exception:
        print("Could not display plot (no display available)")


def main():
    parser = argparse.ArgumentParser(
        description="Test Krisp VIVA filter with real audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python test_audio_filtering.py noisy_input.wav clean_output.wav

  # Adjust noise suppression level
  python test_audio_filtering.py input.wav output.wav --level 80

  # Use different frame duration
  python test_audio_filtering.py input.wav output.wav --frame-duration 20

  # Show spectrograms
  python test_audio_filtering.py input.wav output.wav --visualize

Supported audio formats: WAV, FLAC, OGG, etc. (via soundfile)
Supported sample rates: 8000, 16000, 24000, 32000, 44100, 48000 Hz
        """,
    )

    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("output", help="Output audio file path")
    parser.add_argument(
        "--level",
        type=int,
        default=100,
        help="Noise suppression level (0-100, default: 100)",
    )
    parser.add_argument(
        "--frame-duration",
        type=int,
        default=10,
        choices=[10, 15, 20, 30, 32],
        help="Frame duration in milliseconds (default: 10)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show before/after spectrograms",
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

    # Check model path is set
    if not os.getenv("KRISP_VIVA_FILTER_MODEL_PATH"):
        print("Error: KRISP_VIVA_FILTER_MODEL_PATH environment variable not set")
        print("Set it with: export KRISP_VIVA_FILTER_MODEL_PATH=/path/to/model.kef")
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process the audio
    asyncio.run(
        process_audio_file(
            args.input,
            args.output,
            noise_suppression_level=args.level,
            frame_duration_ms=args.frame_duration,
            show_visualization=args.visualize,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
