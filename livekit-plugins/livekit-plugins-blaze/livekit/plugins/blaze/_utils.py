"""
Internal utilities for livekit-plugins-blaze.

This module contains helper functions used by the plugin implementations.
"""

from __future__ import annotations

from livekit import rtc
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions


def effective_connect_timeout(
    conn_options: APIConnectOptions,
    plugin_timeout: float,
) -> float:
    """Return the plugin timeout when default connect options are used."""
    if not conn_options.timeout:
        return plugin_timeout
    if conn_options is DEFAULT_API_CONNECT_OPTIONS:
        return plugin_timeout
    return conn_options.timeout


def convert_pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 16000,
    num_channels: int = 1,
) -> bytes:
    """Convert raw PCM16 little-endian audio to WAV via ``rtc.AudioFrame``."""
    frame = rtc.AudioFrame(
        data=pcm_data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=len(pcm_data) // (2 * num_channels) if pcm_data else 0,
    )
    return frame.to_wav_bytes()


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
