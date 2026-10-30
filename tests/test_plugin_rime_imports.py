"""Rime TTS imports and actionable errors without the optional STT SDK."""

import subprocess
import sys

import pytest

from livekit.plugins import rime

pytestmark = pytest.mark.unit


def test_tts_import_does_not_require_the_sdk() -> None:
    # A fresh interpreter also catches accidental eager SDK imports from rime.__init__.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from importlib.abc import MetaPathFinder
class NoSDK(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'rimelabs_sdk':
            raise ModuleNotFoundError(name=fullname)
sys.meta_path.insert(0, NoSDK())
from livekit.plugins import rime
assert rime.TTS
try:
    rime.STT()
except ImportError as error:
    assert 'livekit-plugins-rime[stt]' in str(error)
    if sys.version_info < (3, 11):
        assert 'Python 3.11+' in str(error)
else:
    raise AssertionError('Missing SDK was not detected')
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_stt_requires_python_311(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 10, 0))
    with pytest.raises(ImportError, match=r"Rime STT requires Python 3\.11\+"):
        rime.STT()
