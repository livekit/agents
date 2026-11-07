"""Tests for Krisp noise cancellation: backend selection and the suppression level.

Hermetic: they need neither the ``krisp_audio`` wheel nor LiveKit Cloud.
``KrispLicenseAuthProvider`` only checks that its model file exists, so an empty
temporary ``.kef`` file stands in for a model, and the license backend runs on a
faked SDK manager.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

from livekit.plugins.krisp import viva_filter
from livekit.plugins.krisp._krisp import _KrispLicenseFrameProcessor
from livekit.plugins.krisp._sdk import _KrispLicenseSDKManager
from livekit.plugins.krisp.auth import KrispLicenseAuthProvider, LiveKitCloudAuthProvider

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_krisp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KRISP_VIVA_SDK_LICENSE_KEY", raising=False)
    monkeypatch.delenv("KRISP_VIVA_FILTER_MODEL_PATH", raising=False)


@pytest.fixture
def nc_model(tmp_path: Path) -> str:
    path = tmp_path / "nc.kef"
    path.write_bytes(b"")
    return str(path)


@pytest.mark.parametrize("route", ["parameter", "path", "env"])
def test_license_backend_needs_no_key(
    nc_model: str, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    # Krisp's regular builds take no key, so none of the three routes may require one
    monkeypatch.setattr(viva_filter, "_LEGACY_DEPRECATION_SHOWN", True)
    if route == "parameter":
        provider = viva_filter._resolve_auth_provider(
            KrispLicenseAuthProvider(model_path=nc_model), None
        )
    elif route == "path":
        provider = viva_filter._resolve_auth_provider(None, nc_model)
    else:
        monkeypatch.setenv("KRISP_VIVA_FILTER_MODEL_PATH", nc_model)
        provider = viva_filter._resolve_auth_provider(None, None)

    assert isinstance(provider, KrispLicenseAuthProvider)
    assert provider.model_path == nc_model
    assert provider.license_key == ""


def test_env_model_path_takes_the_env_key(nc_model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KRISP_VIVA_FILTER_MODEL_PATH", nc_model)
    monkeypatch.setenv("KRISP_VIVA_SDK_LICENSE_KEY", "api-key")

    provider = viva_filter._resolve_auth_provider(None, None)

    assert isinstance(provider, KrispLicenseAuthProvider)
    assert provider.license_key == "api-key"


def test_env_key_alone_keeps_livekit_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    # turn taking and interruption prediction read the same variable
    monkeypatch.setenv("KRISP_VIVA_SDK_LICENSE_KEY", "api-key")

    assert isinstance(viva_filter._resolve_auth_provider(None, None), LiveKitCloudAuthProvider)


@pytest.mark.parametrize(
    "factory",
    [
        viva_filter.KrispVivaFilterFrameProcessor,
        viva_filter.voice_isolation,
        viva_filter.voice_isolation_telephony,
    ],
)
def test_default_level_is_krisps_100(factory: Callable[..., object]) -> None:
    default = inspect.signature(factory).parameters["noise_suppression_level"].default
    assert isinstance(default, float)
    assert default == 100.0


def test_license_backend_keeps_the_level_as_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_KrispLicenseSDKManager, "acquire", lambda license_key: object())
    monkeypatch.setattr(_KrispLicenseSDKManager, "frame_duration_enum", lambda ms: ms)
    monkeypatch.setattr(_KrispLicenseFrameProcessor, "_create_session", lambda self, rate: None)
    proc = _KrispLicenseFrameProcessor(
        license_key="", model_path="nc.kef", noise_suppression_level=75
    )
    try:
        assert proc.noise_suppression_level == 75.0  # an int is converted
        assert isinstance(proc.noise_suppression_level, float)

        proc.noise_suppression_level = 62.5  # fractions are kept
        assert proc.noise_suppression_level == 62.5
        proc.noise_suppression_level = 150  # clamped to Krisp's range
        assert proc.noise_suppression_level == 100.0
        proc.noise_suppression_level = -5
        assert proc.noise_suppression_level == 0.0
    finally:
        proc._sdk_acquired = False  # nothing to release: acquire() was faked
