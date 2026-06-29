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
import warnings
from typing import Any, Literal, Protocol

from livekit import rtc

from .auth import KrispLicenseAuthProvider, LiveKitCloudAuthProvider

if not hasattr(rtc, "FrameProcessor"):
    raise ImportError(
        "FrameProcessor is not available in your livekit-rtc version. "
        "KrispVivaFilterFrameProcessor requires livekit-rtc >= 1.0.23 with FrameProcessor support. "
        "Please update livekit-rtc: pip install --upgrade 'livekit>=1.0.23'"
    )

_LEGACY_DEPRECATION_SHOWN = False
_FRAME_PARAMS_DEPRECATION_SHOWN = False


class _KrispBackend(Protocol):
    """Structural type for the backend FrameProcessors the facade forwards to.

    Both backends are ``rtc.FrameProcessor`` implementations that additionally
    expose a runtime-settable ``noise_suppression_level`` (not part of the base
    FrameProcessor interface).
    """

    @property
    def enabled(self) -> bool: ...
    @enabled.setter
    def enabled(self, value: bool) -> None: ...
    @property
    def noise_suppression_level(self) -> float: ...
    @noise_suppression_level.setter
    def noise_suppression_level(self, value: float) -> None: ...
    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame: ...
    def _on_credentials_updated(self, *, token: str, url: str) -> None: ...
    def _on_stream_info_updated(
        self, *, room_name: str, participant_identity: str, publication_sid: str
    ) -> None: ...
    def _close(self) -> None: ...


def _resolve_auth_provider(
    auth_provider: LiveKitCloudAuthProvider | KrispLicenseAuthProvider | None,
    model_path: str | None,
) -> LiveKitCloudAuthProvider | KrispLicenseAuthProvider:
    """Pick the auth provider, applying backwards-compat for legacy kwargs.

    Resolution rules:
    1. ``auth_provider`` given → use it.
    2. ``model_path`` given → construct ``KrispLicenseAuthProvider``
       (deprecation warning, since ``model_path`` should be passed on the
       provider instead).
    3. Both ``KRISP_VIVA_SDK_LICENSE_KEY`` and ``KRISP_VIVA_FILTER_MODEL_PATH``
       env vars set → construct ``KrispLicenseAuthProvider`` from the
       environment (the user is responsible for installing ``krisp_audio``).
    4. Otherwise → default ``LiveKitCloudAuthProvider``.
    """
    global _LEGACY_DEPRECATION_SHOWN

    if auth_provider is not None:
        if model_path is not None:
            raise ValueError(
                "Pass model_path on KrispLicenseAuthProvider(model_path=...) "
                "instead of on KrispVivaFilterFrameProcessor when an "
                "auth_provider is supplied."
            )
        return auth_provider

    if model_path is not None:
        if not _LEGACY_DEPRECATION_SHOWN:
            warnings.warn(
                "Passing `model_path` to KrispVivaFilterFrameProcessor is "
                "deprecated. Use `auth_provider=KrispLicenseAuthProvider("
                "model_path=...)` instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            _LEGACY_DEPRECATION_SHOWN = True
        return KrispLicenseAuthProvider(model_path=model_path)

    # Auto-select Krisp-direct license auth when both env vars are set;
    # otherwise fall back to LiveKit Cloud auth.
    if os.getenv("KRISP_VIVA_SDK_LICENSE_KEY") and os.getenv("KRISP_VIVA_FILTER_MODEL_PATH"):
        return KrispLicenseAuthProvider()

    return LiveKitCloudAuthProvider()


def _build_inner(
    provider: LiveKitCloudAuthProvider | KrispLicenseAuthProvider,
    *,
    noise_suppression_level: int,
    frame_duration_ms: int,
    sample_rate: int | None,
) -> _KrispBackend:
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

        cloud_processor: _KrispBackend = _CloudKrispFrameProcessor(
            noise_suppression_level=noise_suppression_level,
            frame_duration_ms=frame_duration_ms,
            sample_rate=sample_rate,
        )
        return cloud_processor

    # KrispLicenseAuthProvider
    from ._krisp import _KrispLicenseFrameProcessor

    return _KrispLicenseFrameProcessor(
        license_key=provider.license_key,
        model_path=provider.model_path,
        noise_suppression_level=noise_suppression_level,
        frame_duration_ms=frame_duration_ms,
        sample_rate=sample_rate,
    )


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
        auth_provider: LiveKitCloudAuthProvider | KrispLicenseAuthProvider | None = None,
        model_path: str | None = None,
        noise_suppression_level: int = 100,
        frame_duration_ms: int | None = None,
        sample_rate: int | None = None,
    ) -> None:
        """Initialize the Krisp frame processor.

        Args:
            noise_suppression_level: Noise suppression level (0-100, default: 100).
            auth_provider: Authentication provider. Defaults to
                :class:`LiveKitCloudAuthProvider` (LiveKit Cloud auth + bundled
                model). Pass :class:`KrispLicenseAuthProvider` to use a Krisp
                license key + ``.kef`` model file directly.
            model_path: **Deprecated.** Use
                ``auth_provider=KrispLicenseAuthProvider(model_path=...)``
                instead. Path to the Krisp model file (``.kef``).
            frame_duration_ms: **Deprecated.** The processor now buffers input
                frames of any size automatically, so this no longer needs to be
                set. Frame duration in milliseconds (10, 15, 20, 30, or 32).
            sample_rate: **Deprecated.** The processor now adapts to the input
                sample rate automatically, so this no longer needs to be set.
                Sample rate in Hz.

        Raises:
            RuntimeError: If the chosen backend wheel is not installed.
            ValueError: If ``frame_duration_ms`` is not supported, or — for
                :class:`KrispLicenseAuthProvider` — if ``model_path`` is
                missing or does not have a ``.kef`` extension.
            FileNotFoundError: If the license-mode model file does not exist.
        """
        if frame_duration_ms is not None or sample_rate is not None:
            global _FRAME_PARAMS_DEPRECATION_SHOWN
            if not _FRAME_PARAMS_DEPRECATION_SHOWN:
                warnings.warn(
                    "Passing `sample_rate` / `frame_duration_ms` to "
                    "KrispVivaFilterFrameProcessor is deprecated. The processor "
                    "now adapts to the input sample rate and frame size "
                    "automatically.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                _FRAME_PARAMS_DEPRECATION_SHOWN = True

        provider = _resolve_auth_provider(auth_provider, model_path)
        self._inner = _build_inner(
            provider,
            noise_suppression_level=noise_suppression_level,
            frame_duration_ms=frame_duration_ms if frame_duration_ms is not None else 10,
            sample_rate=sample_rate,
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
        return self._inner.noise_suppression_level

    @noise_suppression_level.setter
    def noise_suppression_level(self, value: float) -> None:
        """Adjust the noise suppression level (0-100) at runtime."""
        self._inner.noise_suppression_level = value

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
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> Literal[False]:
        """Context manager exit - clean up session."""
        self.close()
        return False
