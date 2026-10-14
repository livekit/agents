"""Public Rime imports must work without loading the optional realtime SDK."""

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit


def _check_imports(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("import_style", ["attribute", "from"])
def test_realtime_namespace_loads_on_demand(import_style: str) -> None:
    pytest.importorskip("rimelabs_sdk.realtime")
    _check_imports(
        f"""
        import importlib
        import sys
        from livekit.plugins import rime

        assert "livekit.plugins.rime.realtime" not in sys.modules
        assert "rimelabs_sdk" not in sys.modules
        assert "realtime" in rime.__all__
        assert "RealtimeModel" not in rime.__all__
        assert not hasattr(rime, "RealtimeModel")

        if {import_style!r} == "from":
            from livekit.plugins.rime import realtime
        else:
            realtime = rime.realtime

        assert realtime is importlib.import_module("livekit.plugins.rime.realtime")
        assert rime.realtime is realtime
        assert callable(realtime.RealtimeModel)
        assert not hasattr(rime, "RealtimeModel")
        """
    )


def test_tts_import_and_missing_realtime_sdk_error() -> None:
    _check_imports(
        """
        import sys
        sys.modules["rimelabs_sdk"] = None

        from livekit.plugins import rime
        from livekit.plugins.rime import TTS

        assert rime.TTS is TTS
        assert "livekit.plugins.rime.realtime" not in sys.modules
        assert not hasattr(rime, "RealtimeModel")
        try:
            rime.realtime
        except ImportError as error:
            assert "livekit-plugins-rime[realtime]" in str(error)
            assert "Python 3.11+" in str(error)
        else:
            raise AssertionError("Realtime imported without its SDK")

        assert "livekit.plugins.rime.realtime" not in sys.modules
        assert "realtime" not in vars(rime)
        assert rime.TTS is TTS
        """
    )
