# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Krisp VIVA noise reduction audio filter for LiveKit Agents.

This module exposes :class:`KrispVivaFilterFrameProcessor`, a thin facade that
forwards to one of two underlying FrameProcessor implementations:

- The closed-source ``krisp_audio_livekit_internal`` wheel (default;
  authenticates via the LiveKit Cloud-managed JWT the agent framework hands
  to FrameProcessors through ``_on_credentials_updated``).
- A local license-mode wrapper around ``krisp_audio`` (selected when the user
  passes :class:`KrispLicenseAuthProvider`).

Note: ``isinstance(processor, SomeBackendInternal)`` will not match — the
public class is the facade, not the backend it forwards to.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import Literal, Protocol

from livekit import rtc
from livekit.plugins.krisp_internal import VivaMode

from .auth import KrispLicenseAuthProvider, LiveKitCloudAuthProvider

if not hasattr(rtc, "FrameProcessor"):
    raise ImportError(
        "FrameProcessor is not available in your livekit-rtc version. "
        "KrispVivaFilterFrameProcessor requires livekit-rtc >= 1.0.23 with FrameProcessor support. "
        "Please update livekit-rtc: pip install --upgrade 'livekit>=1.0.23'"
    )


class _NoiseSuppressionControl(Protocol):
    @property
    def noise_suppression_level(self) -> float: ...
    @noise_suppression_level.setter
    def noise_suppression_level(self, value: float) -> None: ...


def _resolve_auth_provider(
    auth_provider: LiveKitCloudAuthProvider | KrispLicenseAuthProvider | None,
) -> LiveKitCloudAuthProvider | KrispLicenseAuthProvider:
    """Pick the auth provider.

    Resolution rules:
    1. ``auth_provider`` given → use it.
    2. Both ``KRISP_VIVA_SDK_LICENSE_KEY`` and ``KRISP_VIVA_FILTER_MODEL_PATH``
       env vars set → construct ``KrispLicenseAuthProvider`` from the
       environment (the user is responsible for installing ``krisp_audio``).
    3. Otherwise → default ``LiveKitCloudAuthProvider``.
    """
    if auth_provider is not None:
        return auth_provider

    # Auto-select Krisp-direct license auth when both env vars are set;
    # otherwise fall back to LiveKit Cloud auth.
    if os.getenv("KRISP_VIVA_SDK_LICENSE_KEY") and os.getenv("KRISP_VIVA_FILTER_MODEL_PATH"):
        return KrispLicenseAuthProvider()

    return LiveKitCloudAuthProvider()


def _build_inner(
    mode: VivaMode,
    provider: LiveKitCloudAuthProvider | KrispLicenseAuthProvider,
    *,
    noise_suppression_level: int,
) -> tuple[rtc.FrameProcessor[rtc.AudioFrame], _NoiseSuppressionControl]:
    """Lazy-import and construct the backend FrameProcessor."""
    if isinstance(provider, LiveKitCloudAuthProvider):
        try:
            from livekit.plugins.krisp_internal import (
                KrispVivaFilterFrameProcessor as _CloudKrispFrameProcessor,
            )
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "krisp_audio_livekit_internal is missing — this wheel is "
                "bundled with livekit-plugins-krisp, so this likely means a "
                "broken install. Reinstall the plugin "
                "(`pip install --force-reinstall livekit-plugins-krisp`), or "
                "fall back to auth_provider=KrispLicenseAuthProvider(...) if "
                "you have a Krisp license key + .kef model."
            ) from e

        cloud_processor = _CloudKrispFrameProcessor(
            noise_suppression_level=noise_suppression_level,
            frame_duration_ms=10,
            sample_rate=None,
            mode=mode,
        )
        return cloud_processor, cloud_processor

    # KrispLicenseAuthProvider
    from ._krisp import KrispLicenseFrameProcessor

    license_processor = KrispLicenseFrameProcessor(
        license_key=provider.license_key,
        model_path=provider.model_path,
        noise_suppression_level=noise_suppression_level,
        frame_duration_ms=10,
        sample_rate=None,
    )

    return license_processor, license_processor


class KrispVivaFilterFrameProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    """FrameProcessor for Krisp noise reduction.

    Thin facade over two backend FrameProcessor implementations: the
    LiveKit Cloud-bundled wheel (default) and a local wrapper around the
    public ``krisp_audio`` wheel (selected via
    :class:`KrispLicenseAuthProvider`).

    Example:
        ```python
        from livekit.agents import room_io
        from livekit.plugins import krisp

        # Default: uses LiveKit Cloud auth + bundled model.
        processor = krisp.KrispVivaFilterFrameProcessor()

        # Or, explicit Krisp-direct auth with a license + model file.
        processor = krisp.KrispVivaFilterFrameProcessor(
            auth_provider=krisp.auth.krisp_license(
                license_key="...",
                model_path="/path/to/model.kef",
            ),
        )

        await session.start(
            agent=MyAgent(),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    noise_cancellation=processor,
                ),
            ),
        )
        ```
    """

    def __init__(
        self,
        *,
        mode: VivaMode = VivaMode.VOICE_ISOLATION,
        auth_provider: LiveKitCloudAuthProvider | KrispLicenseAuthProvider | None = None,
        noise_suppression_level: int = 75,
    ) -> None:
        """Initialize the Krisp frame processor.

        Args:
            noise_suppression_level: Noise suppression level (0-100, default: 75).
            auth_provider: Authentication provider. Defaults to
                :class:`LiveKitCloudAuthProvider` (LiveKit Cloud auth + bundled
                model). Pass :class:`KrispLicenseAuthProvider` to use a Krisp
                license key + ``.kef`` model file directly. Input frames of any
                size and sample rate are adapted automatically.

        Raises:
            RuntimeError: If the chosen backend wheel is not installed.
            ValueError: For :class:`KrispLicenseAuthProvider`, if ``model_path``
                is missing or does not have a ``.kef`` extension.
            FileNotFoundError: If the license-mode model file does not exist.
        """
        provider = _resolve_auth_provider(auth_provider)
        self._inner, self._noise_control = _build_inner(
            mode,
            provider,
            noise_suppression_level=noise_suppression_level,
        )

    # ----- FrameProcessor hooks: forward to the inner processor -------------

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        return self._inner._process(frame)

    def _on_credentials_updated(self, *, token: str, url: str) -> None:
        self._inner._on_credentials_updated(token=token, url=url)

    def _on_stream_info_updated(
        self,
        *,
        room_name: str,
        participant_identity: str,
        publication_sid: str,
    ) -> None:
        self._inner._on_stream_info_updated(
            room_name=room_name,
            participant_identity=participant_identity,
            publication_sid=publication_sid,
        )

    def _close(self) -> None:
        self._inner._close()

    @property
    def enabled(self) -> bool:
        return self._inner.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._inner.enabled = value

    @property
    def noise_suppression_level(self) -> float:
        """Current noise suppression level (0-100)."""
        return self._noise_control.noise_suppression_level

    @noise_suppression_level.setter
    def noise_suppression_level(self, value: float) -> None:
        """Adjust the noise suppression level (0-100) at runtime."""
        self._noise_control.noise_suppression_level = value

    # ----- Backwards-compat shims ------------------------------------------

    def process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        """Public method that calls _process (for backward compatibility)."""
        return self._process(frame)

    def enable(self) -> None:
        """Enable noise filtering."""
        self.enabled = True

    def disable(self) -> None:
        """Disable noise filtering (audio will pass through unmodified)."""
        self.enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if filtering is currently enabled (backward compatibility)."""
        return self.enabled

    def close(self) -> None:
        """Clean up processor session resources (public method for backward compatibility)."""
        self._close()

    def __enter__(self) -> KrispVivaFilterFrameProcessor:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Context manager exit - clean up session."""
        self.close()
        return False


def voice_isolation(
    *,
    auth_provider: LiveKitCloudAuthProvider | KrispLicenseAuthProvider | None = None,
    noise_suppression_level: int = 75,
) -> KrispVivaFilterFrameProcessor:
    """Create a Krisp VIVA voice isolation ``FrameProcessor``.

    Pass the result as the ``noise_cancellation`` argument in
    :class:`~livekit.agents.room_io.AudioInputOptions`. Uses LiveKit Cloud auth
    by default; pass ``auth_provider=krisp.auth.krisp_license(...)`` to run the
    proprietary Krisp SDK with your own license key + ``.kef`` model.

    Args:
        auth_provider: Authentication provider. Defaults to
            :class:`LiveKitCloudAuthProvider` (LiveKit Cloud auth + bundled model).
        noise_suppression_level: Noise suppression level (0-100, default: 75).
    """
    return KrispVivaFilterFrameProcessor(
        mode=VivaMode.VOICE_ISOLATION,
        auth_provider=auth_provider,
        noise_suppression_level=noise_suppression_level,
    )


def voice_isolation_telephony(
    *,
    auth_provider: LiveKitCloudAuthProvider | KrispLicenseAuthProvider | None = None,
    noise_suppression_level: int = 75,
) -> KrispVivaFilterFrameProcessor:
    """Create a Krisp VIVA voice isolation ``FrameProcessor`` tuned for telephony.

    Same as :func:`voice_isolation`, but optimized for telephony audio such as
    SIP participants.

    Args:
        auth_provider: Authentication provider. Defaults to
            :class:`LiveKitCloudAuthProvider` (LiveKit Cloud auth + bundled model).
        noise_suppression_level: Noise suppression level (0-100, default: 75).
    """
    return KrispVivaFilterFrameProcessor(
        mode=VivaMode.VOICE_ISOLATION_TELEPHONY,
        auth_provider=auth_provider,
        noise_suppression_level=noise_suppression_level,
    )
