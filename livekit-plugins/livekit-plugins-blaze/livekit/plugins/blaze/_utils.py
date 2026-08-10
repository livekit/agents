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
    """Return the plugin timeout when framework default connect options are used.

    Compares the timeout *value* (not object identity) so framework-copied
    options — e.g. ``DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS`` or
    ``APIConnectOptions`` rebuilt by the voice pipeline — still honour
    plugin-configured ``stt_timeout`` / ``tts_timeout`` / ``llm_timeout``.
    Explicit non-default timeouts on ``conn_options`` still win.
    """
    if not conn_options.timeout:
        return plugin_timeout
    # Treat the framework default timeout as "unset" regardless of whether
    # the caller passed the DEFAULT_API_CONNECT_OPTIONS singleton or a copy.
    if conn_options.timeout == DEFAULT_API_CONNECT_OPTIONS.timeout:
        return plugin_timeout
    return conn_options.timeout


def _host_from_authority(authority: str) -> str:
    """Extract hostname from ``host``, ``host:port``, or ``[IPv6]:port``."""
    authority = authority.split("/")[0]
    if "@" in authority:
        authority = authority.rsplit("@", 1)[-1]
    if authority.startswith("["):
        end = authority.find("]")
        if end != -1:
            return authority[1:end]
    # IPv4 or hostname — strip :port (not valid bare IPv6 without brackets)
    return authority.split(":")[0]


def _strip_userinfo(rest: str) -> str:
    """Remove ``userinfo@`` from a scheme-less URL rest (``auth@host/path``).

    Auth is sent on the first WebSocket frame, so embedded URL credentials
    must not redirect the connection (or the bearer token) to an unexpected
    authority. Only the host/port/path are kept.
    """
    if "@" not in rest:
        return rest
    if "/" in rest:
        authority, _, path = rest.partition("/")
        if "@" in authority:
            authority = authority.rsplit("@", 1)[-1]
        return f"{authority}/{path}" if path else authority
    return rest.rsplit("@", 1)[-1]


def _is_loopback_host(host: str) -> bool:
    """True for hosts safe to use plaintext ``ws://`` (token on first frame).

    Only true loopback addresses qualify. mDNS / LAN names ending in
    ``.local`` are *not* treated as local — the bearer token must not cross
    the network in cleartext.
    """
    return host in ("localhost", "127.0.0.1", "::1")


def ws_base_url(api_url: str) -> str:
    """Convert an HTTP(S) API base URL to a WebSocket base URL.

    Auth tokens are sent on the first STT/TTS WS message, so non-loopback
    endpoints always use ``wss://`` (even if ``BLAZE_API_URL`` is ``http://``).
    Only true loopback hosts (``localhost`` / ``127.0.0.1`` / ``::1``) keep
    plaintext ``ws://`` for local development.

    Embedded URL userinfo (``user@host`` / ``user:pass@host``) is stripped so
    a misconfigured ``BLAZE_API_URL`` cannot redirect the bearer token.
    """
    base = api_url.strip().rstrip("/")
    if base.startswith("https://"):
        return "wss://" + _strip_userinfo(base[len("https://") :])
    if base.startswith("wss://"):
        return "wss://" + _strip_userinfo(base[len("wss://") :])
    if base.startswith("ws://"):
        rest = _strip_userinfo(base[len("ws://") :])
        if _is_loopback_host(_host_from_authority(rest)):
            return "ws://" + rest
        return "wss://" + rest
    if base.startswith("http://"):
        rest = _strip_userinfo(base[len("http://") :])
        if _is_loopback_host(_host_from_authority(rest)):
            return "ws://" + rest
        return "wss://" + rest
    # No scheme — assume HTTPS/WSS
    return "wss://" + _strip_userinfo(base.lstrip("/"))


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
