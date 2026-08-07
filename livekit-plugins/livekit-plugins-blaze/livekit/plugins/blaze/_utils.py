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


def ws_base_url(api_url: str) -> str:
    """Convert an HTTP(S) API base URL to a WebSocket base URL.

    Auth tokens are sent on the first STT/TTS WS message, so remote endpoints
    always use ``wss://`` (even if ``BLAZE_API_URL`` is ``http://``). Local
    development hosts keep ``ws://``.
    """
    base = api_url.strip().rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :]
    if base.startswith("wss://"):
        return base
    if base.startswith("ws://"):
        host = base[len("ws://") :].split("/")[0].split(":")[0]
        if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local"):
            return base
        return "wss://" + base[len("ws://") :]
    if base.startswith("http://"):
        rest = base[len("http://") :]
        host = rest.split("/")[0].split(":")[0]
        if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local"):
            return "ws://" + rest
        return "wss://" + rest
    # No scheme — assume HTTPS/WSS
    return "wss://" + base.lstrip("/")


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
