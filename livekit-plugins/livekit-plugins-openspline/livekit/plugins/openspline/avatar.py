# SPDX-FileCopyrightText: 2026 OpenSpline contributors
# SPDX-License-Identifier: Apache-2.0
"""Provider configuration for native OpenSpline avatar output."""

from pathlib import Path

from openspline import Openspline
from openspline.config import Quality
from openspline.integrations.livekit import AvatarSession as NativeAvatarSession


class AvatarSession(NativeAvatarSession):
    """Give an existing LiveKit agent a face in OpenSpline Cloud or on your GPU.

    Set OPENSPLINE_API_KEY and provide avatar_id for a saved Cloud avatar, or
    provide portrait as a path or bytes. Without a Cloud key the default service
    is http://localhost:7860. Start before AgentSession.start; start waits for
    the avatar video. The job owns cleanup; outside a job call aclose().

    Args:
        portrait: Local image path or bytes; mutually exclusive with avatar_id.
        avatar_id: Owned saved Cloud avatar ID, copied from /developers.
        api_key: OpenSpline Cloud key; defaults to OPENSPLINE_API_KEY.
        url: OpenSpline service URL; defaults to OPENSPLINE_URL or SDK defaults.
        quality: low (default) or high; an injected client's default is respected.
        background: Optional background image/video path or bytes.
        client: Existing OpenSpline client; cannot be combined with api_key/url.
        livekit_url: LiveKit server URL; defaults to LIVEKIT_URL.
        livekit_api_key: LiveKit key; defaults to LIVEKIT_API_KEY.
        livekit_api_secret: LiveKit secret; defaults to LIVEKIT_API_SECRET.
        join_timeout: Seconds to wait for video after worker preparation (30).
    """

    def __init__(
        self,
        *,
        portrait: str | Path | bytes | None = None,
        avatar_id: str | None = None,
        api_key: str | None = None,
        url: str | None = None,
        quality: Quality | None = None,
        background: str | Path | bytes | None = None,
        client: Openspline | None = None,
        livekit_url: str | None = None,
        livekit_api_key: str | None = None,
        livekit_api_secret: str | None = None,
        join_timeout: float = 30.0,
    ) -> None:
        if client is not None and (api_key is not None or url is not None):
            raise ValueError("Pass client or api_key/url, not both")
        super().__init__(
            portrait,
            quality=quality,
            client=client or Openspline(api_key=api_key, url=url),
            avatar_id=avatar_id,
            background=background,
            url=livekit_url,
            api_key=livekit_api_key,
            api_secret=livekit_api_secret,
            join_timeout=join_timeout,
        )
