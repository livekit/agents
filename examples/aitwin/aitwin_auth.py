"""Mint short-lived AiTwin bearer tokens for server-side API calls (alignVisemes)."""

from __future__ import annotations

import logging
import os
import time
from uuid import uuid4

import requests

logger = logging.getLogger("aitwin-livekit-demo.auth")

_cached_token: str | None = None
_cached_expires_at: float = 0.0


def mint_aitwin_auth_token(*, force: bool = False) -> str:
    """
    Return a bearer token for AITWIN_API_BASE.
    Reuses a cached token until ~60s before expiry.
    """
    global _cached_token, _cached_expires_at

    now = time.time()
    if (
        not force
        and _cached_token
        and now < _cached_expires_at - 60
    ):
        return _cached_token

    client_id = (os.getenv("AITWIN_CLIENT_ID") or "").strip()
    client_secret = (os.getenv("AITWIN_CLIENT_SECRET") or "").strip()
    auth_url = (os.getenv("AITWIN_AUTH_TOKEN_URL") or "").strip()
    if not client_id or not client_secret or not auth_url:
        raise RuntimeError(
            "AITWIN_CLIENT_ID / AITWIN_CLIENT_SECRET / AITWIN_AUTH_TOKEN_URL required "
            "for /avatar_alignVisemes"
        )

    expires_in = 60 * 60
    response = requests.post(
        auth_url,
        headers={
            "Content-Type": "application/json",
            "Client-Id": client_id,
            "Client-Secret": client_secret,
        },
        json={
            "userId": f"agent-{uuid4().hex[:12]}",
            "userName": "livekit-agent",
            "expiresIn": expires_in,
        },
        timeout=20,
    )
    data = response.json() if response.content else {}
    if not response.ok or not data.get("success") or not data.get("authToken"):
        raise RuntimeError(
            data.get("error") or f"AiTwin auth failed ({response.status_code})"
        )

    token = str(data["authToken"])
    _cached_token = token
    _cached_expires_at = now + expires_in
    logger.info("minted AiTwin auth token (expires_in=%ss)", expires_in)
    return token
