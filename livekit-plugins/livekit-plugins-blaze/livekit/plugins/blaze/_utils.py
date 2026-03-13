"""
Internal utilities for livekit-plugins-blaze.

This module contains helper functions used by the plugin implementations.
"""

import io
import struct


def convert_pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Convert raw PCM audio data to WAV format with proper headers.

    Args:
        pcm_data: Raw PCM audio data (16-bit signed integers, little-endian)
        sample_rate: Audio sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1 for mono)
        bits_per_sample: Bits per sample (default: 16)

    Returns:
        WAV format audio data with RIFF header, fmt chunk, and data chunk
    """
    pcm_size = len(pcm_data)

    # Calculate WAV file structure
    # "WAVE" + fmt chunk (24 bytes) + data chunk header (8 bytes) + data
    file_size = 4 + 24 + 8 + pcm_size

    wav_buffer = io.BytesIO()

    # RIFF header
    wav_buffer.write(b"RIFF")
    wav_buffer.write(struct.pack("<I", file_size))
    wav_buffer.write(b"WAVE")

    # fmt chunk
    wav_buffer.write(b"fmt ")
    wav_buffer.write(struct.pack("<I", 16))  # Chunk size (16 bytes for PCM)
    wav_buffer.write(struct.pack("<H", 1))  # Audio format (1 = PCM)
    wav_buffer.write(struct.pack("<H", channels))
    wav_buffer.write(struct.pack("<I", sample_rate))
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    wav_buffer.write(struct.pack("<I", byte_rate))
    block_align = channels * (bits_per_sample // 8)
    wav_buffer.write(struct.pack("<H", block_align))
    wav_buffer.write(struct.pack("<H", bits_per_sample))

    # data chunk
    wav_buffer.write(b"data")
    wav_buffer.write(struct.pack("<I", pcm_size))
    wav_buffer.write(pcm_data)

    return wav_buffer.getvalue()


def apply_normalization_rules(
    text: str,
    rules: dict[str, str] | None,
) -> str:
    """
    Apply text normalization rules.

    Performs simple string replacement based on the provided rules dictionary.
    Matching is case-sensitive.

    Args:
        text: Input text to normalize
        rules: Dictionary mapping patterns to their replacements.
               If None or empty, returns original text.

    Returns:
        Text with all matching patterns replaced
    """
    if not rules:
        return text

    # Apply longer patterns first for more deterministic results.
    # Example: {"USD": "...", "$": "..."} should replace "USD" before "$".
    result = text
    for pattern, replacement in sorted(rules.items(), key=lambda kv: len(kv[0]), reverse=True):
        if not pattern:
            continue
        result = result.replace(pattern, replacement)
    return result
