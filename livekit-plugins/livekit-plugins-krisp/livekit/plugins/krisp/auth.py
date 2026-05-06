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

"""Authentication providers for the Krisp plugin.

A provider determines which Krisp backend module is loaded and how it is
authenticated:

- :class:`LiveKitCloudAuthProvider` (default): uses the closed-source
  ``krisp_audio_livekit_internal`` wheel. The wheel bundles the model and
  authenticates against LiveKit Cloud using the room's JWT, which the agent
  framework hands to the FrameProcessor via the standard
  ``_on_credentials_updated`` callback (and refreshes automatically on
  ``token_refreshed``).
- :class:`KrispLicenseAuthProvider`: uses the public ``krisp_audio`` wheel
  with a Krisp license key and a ``.kef`` model file.

Only the chosen backend module is imported.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .log import logger


@dataclass
class KrispBackend:
    """Handle for a loaded Krisp SDK module.

    Returned by :meth:`KrispAuthProvider.init_sdk`. Encapsulates the imported
    module so that callers (the frame processor) never reference
    ``krisp_audio`` or ``krisp_audio_livekit_internal`` directly.
    """

    module: Any
    name: str
    sample_rates: dict[int, Any] = field(default_factory=dict)
    frame_durations: dict[int, Any] = field(default_factory=dict)
    # True once a session can be created. License backends are ready
    # immediately after init_sdk; cloud backends are ready only after
    # credentials arrive.
    ready: bool = True

    def to_sample_rate_enum(self, sample_rate: int) -> Any:
        if sample_rate not in self.sample_rates:
            supported = ", ".join(str(r) for r in sorted(self.sample_rates))
            raise ValueError(
                f"Unsupported sample rate: {sample_rate} Hz. Supported rates: {supported} Hz"
            )
        return self.sample_rates[sample_rate]

    def to_frame_duration_enum(self, frame_duration_ms: int) -> Any:
        if frame_duration_ms not in self.frame_durations:
            supported = ", ".join(str(d) for d in sorted(self.frame_durations))
            raise ValueError(
                f"Unsupported frame duration: {frame_duration_ms} ms. "
                f"Supported durations: {supported} ms"
            )
        return self.frame_durations[frame_duration_ms]

    def create_session(
        self,
        sample_rate: int,
        frame_duration_ms: int,
        *,
        model_path: str | None,
    ) -> Any:
        """Create a Krisp NC session for the given audio format.

        ``model_path`` is required for the license backend and ignored for the
        LiveKit Cloud backend (which bundles its model).
        """
        if not self.ready:
            raise RuntimeError(
                "Krisp backend is not yet ready (waiting for credentials from "
                "the agent framework). Sessions are created lazily on the first "
                "_on_credentials_updated callback."
            )
        cfg = self.module.NcSessionConfig()
        cfg.inputSampleRate = self.to_sample_rate_enum(sample_rate)
        cfg.inputFrameDuration = self.to_frame_duration_enum(frame_duration_ms)
        cfg.outputSampleRate = cfg.inputSampleRate

        if model_path is not None:
            model_info = self.module.ModelInfo()
            model_info.path = model_path
            cfg.modelInfo = model_info

        return self.module.NcInt16.create(cfg)

    def version(self) -> tuple[int, int, int]:
        v = self.module.getVersion()
        return v.major, v.minor, v.patch


def _build_enum_dicts(module: Any) -> tuple[dict[int, Any], dict[int, Any]]:
    sample_rates = {
        8000: module.SamplingRate.Sr8000Hz,
        16000: module.SamplingRate.Sr16000Hz,
        24000: module.SamplingRate.Sr24000Hz,
        32000: module.SamplingRate.Sr32000Hz,
        44100: module.SamplingRate.Sr44100Hz,
        48000: module.SamplingRate.Sr48000Hz,
    }
    frame_durations = {
        10: module.FrameDuration.Fd10ms,
        15: module.FrameDuration.Fd15ms,
        20: module.FrameDuration.Fd20ms,
        30: module.FrameDuration.Fd30ms,
        32: module.FrameDuration.Fd32ms,
    }
    return sample_rates, frame_durations


class KrispAuthProvider(ABC):
    """Abstract base for Krisp SDK authentication providers."""

    name: ClassVar[str]
    # When True, the FrameProcessor must wait for the framework to push
    # credentials via _on_credentials_updated before sessions can be created
    # or audio can be processed.
    needs_runtime_credentials: ClassVar[bool] = False

    @abstractmethod
    def init_sdk(self) -> KrispBackend:
        """Import the backend module and return a handle.

        For providers with ``needs_runtime_credentials = True``, this should
        only load the module and return a backend with ``ready=False``; the
        actual underlying ``init(...)`` call is deferred to
        :meth:`update_credentials`.
        """

    @abstractmethod
    def destroy_sdk(self, backend: KrispBackend) -> None:
        """Tear down the backend. Called when the last reference is released."""

    def update_credentials(  # noqa: B027 — intentional no-op hook
        self, backend: KrispBackend, *, token: str, url: str
    ) -> None:
        """Forward a fresh JWT + url to the backend.

        Called by :class:`KrispVivaFilterFrameProcessor` from
        ``_on_credentials_updated`` (initial track subscription) and on every
        ``token_refreshed`` event from the room. Default: no-op (license
        backends authenticate up-front and don't need runtime credentials).
        """

    def update_stream_info(  # noqa: B027 — intentional no-op hook
        self,
        backend: KrispBackend,
        *,
        room_name: str,
        participant_identity: str,
        publication_sid: str,
    ) -> None:
        """Forward stream metadata to the backend.

        Called by the FrameProcessor from ``_on_stream_info_updated`` when a
        track becomes available or switches. Default: no-op.
        """


class LiveKitCloudAuthProvider(KrispAuthProvider):
    """LiveKit Cloud-managed Krisp auth.

    Uses the closed-source ``krisp_audio_livekit_internal`` wheel. The wheel
    bundles the noise-suppression model, so no Krisp env vars are required.
    Authentication and metering happen via the room's existing JWT, which the
    agent framework pushes to the FrameProcessor through the standard
    ``_on_credentials_updated`` callback (and refreshes automatically when the
    room raises ``token_refreshed``).

    The provider takes no constructor arguments — credentials come from the
    framework, not from the user.
    """

    name: ClassVar[str] = "livekit_cloud"
    needs_runtime_credentials: ClassVar[bool] = True

    def __init__(self) -> None:
        self._initialized = False

    def init_sdk(self) -> KrispBackend:
        try:
            import krisp_audio_livekit_internal as krisp_module  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "krisp_audio_livekit_internal is missing — this wheel is "
                "bundled with livekit-plugins-krisp, so this likely means a "
                "broken install. Reinstall the plugin "
                "(`pip install --force-reinstall livekit-plugins-krisp`), or "
                "fall back to auth_provider=KrispLicenseAuthProvider(...) if "
                "you have a Krisp license key + .kef model."
            ) from e

        sample_rates, frame_durations = _build_enum_dicts(krisp_module)
        # Backend is not yet ready — the underlying krisp init(token, url) call
        # is deferred until the framework pushes credentials.
        return KrispBackend(
            module=krisp_module,
            name=self.name,
            sample_rates=sample_rates,
            frame_durations=frame_durations,
            ready=False,
        )

    def destroy_sdk(self, backend: KrispBackend) -> None:
        if self._initialized:
            backend.module.destroy()
            self._initialized = False

    def update_credentials(self, backend: KrispBackend, *, token: str, url: str) -> None:
        if not self._initialized:
            backend.module.init(token=token, url=url)
            self._initialized = True
            backend.ready = True
            version = backend.version()
            logger.debug(
                "Krisp Audio SDK (LiveKit Cloud) initialized - Version: %d.%d.%d",
                version[0],
                version[1],
                version[2],
            )
        else:
            # Token rotation. The exact API on krisp_audio_livekit_internal is
            # subject to confirmation; expected to be a no-arg-checked
            # credential update separate from full re-init.
            backend.module.update_credentials(token=token, url=url)
            logger.debug("Krisp Audio SDK (LiveKit Cloud) credentials refreshed")


class KrispLicenseAuthProvider(KrispAuthProvider):
    """Krisp-direct auth using a license key and a ``.kef`` model file.

    Defaults to reading ``KRISP_VIVA_SDK_LICENSE_KEY`` and
    ``KRISP_VIVA_FILTER_MODEL_PATH`` from the environment.
    """

    name: ClassVar[str] = "license"
    needs_runtime_credentials: ClassVar[bool] = False

    def __init__(
        self,
        *,
        license_key: str | None = None,
        model_path: str | None = None,
    ) -> None:
        self._license_key = license_key or os.getenv("KRISP_VIVA_SDK_LICENSE_KEY", "")
        self._model_path = model_path or os.getenv("KRISP_VIVA_FILTER_MODEL_PATH")

        if not self._model_path:
            raise ValueError(
                "Krisp model path is required. Pass model_path=... or set "
                "KRISP_VIVA_FILTER_MODEL_PATH."
            )
        if not self._model_path.endswith(".kef"):
            raise ValueError("Krisp model must have .kef extension")
        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(f"Krisp model file not found: {self._model_path}")

    @property
    def model_path(self) -> str:
        assert self._model_path is not None
        return self._model_path

    def init_sdk(self) -> KrispBackend:
        try:
            import krisp_audio as krisp_module  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "krisp-audio is not installed. Install Krisp's wheel "
                "(`pip install krisp-audio`) or use the default "
                "LiveKitCloudAuthProvider."
            ) from e

        krisp_module.globalInit(
            "",
            self._license_key,
            self._licensing_error_callback,
            self._log_callback,
            krisp_module.LogLevel.Off,
        )

        sample_rates, frame_durations = _build_enum_dicts(krisp_module)
        backend = KrispBackend(
            module=krisp_module,
            name=self.name,
            sample_rates=sample_rates,
            frame_durations=frame_durations,
            ready=True,
        )
        version = backend.version()
        logger.debug(
            "Krisp Audio SDK (license) initialized - Version: %d.%d.%d",
            version[0],
            version[1],
            version[2],
        )
        return backend

    def destroy_sdk(self, backend: KrispBackend) -> None:
        backend.module.globalDestroy()

    @staticmethod
    def _log_callback(log_message: str, log_level: Any) -> None:
        logger.debug(f"[Krisp {log_level}] {log_message}")

    @staticmethod
    def _licensing_error_callback(error: Any, error_message: str) -> None:
        logger.error(f"[Krisp Licensing Error: {error}] {error_message}")
