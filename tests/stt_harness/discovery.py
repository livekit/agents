"""Load provider-owned contracts without importing their production plugins."""

import importlib.util
import sys
from pathlib import Path

from .protocol import Provider, SDKProvider


def protocol_modules():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "livekit-plugins").glob("*/tests/stt_protocol.py"))
    paths += [root / "livekit-agents/tests/inference/stt_protocol.py"]
    for index, path in enumerate(paths):
        spec = importlib.util.spec_from_file_location(f"stt_protocol_{index}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield path, module


def providers() -> tuple[Provider | SDKProvider, ...]:
    entries = tuple(provider for _, module in protocol_modules() for provider in module.PROVIDERS)
    assert len({provider.name for provider in entries}) == len(entries), (
        "duplicate STT provider name"
    )
    return entries


def websocket_providers() -> tuple[Provider, ...]:
    return tuple(provider for provider in providers() if isinstance(provider, Provider))


def sdk_providers() -> tuple[SDKProvider, ...]:
    return tuple(provider for provider in providers() if isinstance(provider, SDKProvider))


def provider_by_name(name: str) -> Provider | SDKProvider:
    return next(provider for provider in providers() if provider.name == name)


def transport_fixtures():
    return tuple(
        provider.transport_fixture
        for provider in providers()
        if provider.transport_fixture is not None
    )
