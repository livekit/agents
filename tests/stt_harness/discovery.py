"""Load provider-owned contracts without importing their production plugins."""

import importlib.util
import sys
from pathlib import Path

from .protocol import Provider, SDKProvider


def modules():
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
    cases = tuple(case for _, module in modules() for case in module.PROVIDERS)
    assert len({case.name for case in cases}) == len(cases), "duplicate STT provider name"
    return cases


def websocket_providers() -> tuple[Provider, ...]:
    return tuple(case for case in providers() if isinstance(case, Provider))


def sdk_providers() -> tuple[SDKProvider, ...]:
    return tuple(case for case in providers() if isinstance(case, SDKProvider))


def provider_by_name(name: str) -> Provider | SDKProvider:
    return next(case for case in providers() if case.name == name)


def fault_cases():
    return tuple(case.fault_case for case in providers() if case.fault_case is not None)
