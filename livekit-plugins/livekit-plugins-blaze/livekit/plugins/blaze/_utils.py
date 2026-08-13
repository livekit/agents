"""
Internal utilities for livekit-plugins-blaze.

This module contains helper functions used by the plugin implementations.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from livekit import rtc
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

# Schemes accepted for BLAZE_API_URL / WebSocket base URLs.
_ALLOWED_API_SCHEMES = frozenset({"http", "https", "ws", "wss"})
_HTTP_API_SCHEMES = frozenset({"http", "https"})

# Redact obvious credential material that a misbehaving gateway might echo.
_BEARER_RE = re.compile(r"(?i)\bBearer\s+\S+")
_JSON_TOKEN_RE = re.compile(r'(?i)("token"\s*:\s*")[^"]*(")')


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


def _is_loopback_host(host: str) -> bool:
    """True for hosts safe to use plaintext ``ws://`` (token on first frame).

    Only true loopback addresses qualify. mDNS / LAN names ending in
    ``.local`` are *not* treated as local — the bearer token must not cross
    the network in cleartext.
    """
    return host in ("localhost", "127.0.0.1", "::1")


def _authority_without_userinfo(hostname: str, port: int | None) -> str:
    """Build ``host``, ``host:port``, or ``[IPv6]:port`` without userinfo."""
    # urlsplit hostname is unbracketed; re-bracket IPv6 for the netloc.
    if ":" in hostname:
        host_part = f"[{hostname}]"
    else:
        host_part = hostname
    if port is not None:
        return f"{host_part}:{port}"
    return host_part


def validate_api_base_url(api_url: str, *, http_only: bool = False) -> str:
    """Validate and normalize a Blaze API/WS base URL.

    Requires an absolute URL with an allowed scheme and a non-empty hostname.
    Path-only or scheme-less values are rejected so a misconfigured
    ``BLAZE_API_URL`` cannot silently become a bad ``wss://`` target that
    still carries the bearer token.

    Args:
        api_url: Candidate base URL (trailing slashes are stripped).
        http_only: When True (HTTP API base), only ``http``/``https`` are allowed.

    Returns:
        Normalized URL without a trailing slash.

    Raises:
        ValueError: If the URL is empty, lacks a scheme/host, or uses a
            disallowed scheme.
    """
    base = api_url.strip().rstrip("/")
    if not base:
        raise ValueError("Blaze API URL must be a non-empty absolute http(s) URL")

    parsed = urlsplit(base)
    scheme = (parsed.scheme or "").lower()
    allowed = _HTTP_API_SCHEMES if http_only else _ALLOWED_API_SCHEMES
    if scheme not in allowed:
        raise ValueError(
            "Blaze API URL must be an absolute "
            f"{'http(s)' if http_only else 'http(s) or ws(s)'} URL with a hostname, "
            f"got {api_url!r}"
        )
    host = parsed.hostname
    if not host:
        raise ValueError(
            f"Blaze API URL must include a non-empty hostname, got {api_url!r}"
        )

    authority = _authority_without_userinfo(host, parsed.port)
    path = parsed.path.rstrip("/") if parsed.path else ""
    # Drop query/fragment and userinfo — auth is sent via token, not the URL.
    return f"{scheme}://{authority}{path}"


def ws_base_url(api_url: str) -> str:
    """Convert an HTTP(S) API base URL to a WebSocket base URL.

    Auth tokens are sent on the first STT/TTS WS message, so non-loopback
    endpoints always use ``wss://`` (even if ``BLAZE_API_URL`` is ``http://``).
    Only true loopback hosts (``localhost`` / ``127.0.0.1`` / ``::1``) keep
    plaintext ``ws://`` for local development.

    The input must be an absolute http(s)/ws(s) URL with a hostname.
    Embedded URL userinfo is stripped so a misconfigured ``BLAZE_API_URL``
    cannot redirect the bearer token.
    """
    base = validate_api_base_url(api_url, http_only=False)
    parsed = urlsplit(base)
    scheme = (parsed.scheme or "").lower()
    host = parsed.hostname or ""
    authority = _authority_without_userinfo(host, parsed.port)
    path = parsed.path or ""

    if scheme in ("https", "wss"):
        ws_scheme = "wss"
    elif _is_loopback_host(host):
        ws_scheme = "ws"
    else:
        # http:// or ws:// on a remote host — force TLS (token on first frame).
        ws_scheme = "wss"

    return f"{ws_scheme}://{authority}{path}"


def redact_secrets(text: str, token: str | None = None) -> str:
    """Redact known auth tokens and bearer-looking strings from error text."""
    if not text:
        return text
    if token:
        text = text.replace(token, "***")
    text = _BEARER_RE.sub("Bearer ***", text)
    text = _JSON_TOKEN_RE.sub(r"\1***\2", text)
    return text


def safe_ws_error_detail(
    msg: Any,
    *,
    token: str | None = None,
    max_text_len: int = 200,
) -> str:
    """Build a short, secret-safe description of a WebSocket server frame.

    Avoids interpolating the full raw message (which may echo the auth token
    sent on the first frame). Prefers type/status + a truncated text field.
    """
    if isinstance(msg, dict):
        code = msg.get("type") or msg.get("status") or msg.get("code") or "error"
        raw_text = msg.get("text") or msg.get("message") or msg.get("details") or ""
        if not isinstance(raw_text, str):
            raw_text = str(raw_text) if raw_text else ""
        text = redact_secrets(raw_text, token).strip()
        if len(text) > max_text_len:
            text = text[:max_text_len] + "..."
        if text:
            return f"{code}: {text}"
        return str(code)

    text = redact_secrets(str(msg), token).strip()
    if len(text) > max_text_len:
        text = text[:max_text_len] + "..."
    return text or "error"


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
