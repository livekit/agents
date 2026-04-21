# test_install.py — smoke test for pip install -e plugin/
import subprocess
import sys


def test_pip_metadata():
    """Package metadata is correct (name, version, editable location)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "livekit-plugins-60db"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("[FAIL] test_pip_metadata: 'pip show livekit-plugins-60db' returned non-zero")
            return False

        lines = result.stdout.strip().splitlines()
        meta = {}
        for line in lines:
            if ": " in line:
                k, v = line.split(": ", 1)
                meta[k] = v

        name = meta.get("Name", "")
        version = meta.get("Version", "")
        location = meta.get("Editable project location", meta.get("Location", ""))

        errors = []
        if name != "livekit-plugins-60db":
            errors.append(f"expected Name='livekit-plugins-60db', got '{name}'")
        if not version:
            errors.append("Version is missing")
        if not location:
            errors.append("Editable project location is missing")

        if errors:
            print(f"[FAIL] test_pip_metadata: {'; '.join(errors)}")
            return False

        print(f"[PASS] test_pip_metadata: name={name}, version={version}, location={location}")
        return True
    except Exception as e:
        print(f"[FAIL] test_pip_metadata: {e}")
        return False


def test_imports():
    """All __all__ exports are importable."""
    try:
        from livekit.plugins import _60db

        expected = ["_60dbClient", "LLM", "STT", "TTS", "__version__"]
        for name in expected:
            if not hasattr(_60db, name):
                print(f"[FAIL] test_imports: {name} not found in livekit.plugins._60db")
                return False

        print("[PASS] test_imports: _60dbClient, LLM, STT, TTS, __version__ all imported")
        return True
    except ImportError as e:
        print(f"[FAIL] test_imports: {e}")
        return False


def test_version():
    """__version__ matches version.py."""
    try:
        from livekit.plugins._60db import __version__
        from livekit.plugins._60db.version import __version__ as file_version

        if __version__ != file_version:
            print(f"[FAIL] test_version: __init__.__version__={__version__} != version.py={file_version}")
            return False

        print(f"[PASS] test_version: __version__={__version__}")
        return True
    except Exception as e:
        print(f"[FAIL] test_version: {e}")
        return False


def test_plugin_registration():
    """SixtyDbPlugin is registered with livekit.agents.Plugin."""
    try:
        from livekit.agents import Plugin

        titles = [p.title for p in Plugin.registered_plugins]
        if "livekit.plugins._60db" not in titles:
            print(f"[FAIL] test_plugin_registration: 'livekit.plugins._60db' not in registered plugins {titles}")
            return False

        print("[PASS] test_plugin_registration: 'livekit.plugins._60db' found in registered plugins")
        return True
    except Exception as e:
        print(f"[FAIL] test_plugin_registration: {e}")
        return False


def test_instantiation():
    """TTS(), STT(), LLM() can be created with an API key (no network calls)."""
    try:
        from livekit.plugins._60db import LLM, STT, TTS

        tts = TTS(api_key="test-key")
        stt = STT(api_key="test-key")
        llm = LLM(api_key="test-key")

        print(f"[PASS] test_instantiation: TTS={type(tts).__name__}, STT={type(stt).__name__}, LLM={type(llm).__name__}")
        return True
    except Exception as e:
        print(f"[FAIL] test_instantiation: {e}")
        return False


def test_class_hierarchy():
    """TTS/STT/LLM inherit from the correct LiveKit base classes."""
    try:
        from livekit.agents import llm as lk_llm, stt as lk_stt, tts as lk_tts
        from livekit.plugins._60db import LLM, STT, TTS

        errors = []
        if not issubclass(TTS, lk_tts.TTS):
            errors.append("TTS does not inherit from livekit.agents.tts.TTS")
        if not issubclass(STT, lk_stt.STT):
            errors.append("STT does not inherit from livekit.agents.stt.STT")
        if not issubclass(LLM, lk_llm.LLM):
            errors.append("LLM does not inherit from livekit.agents.llm.LLM")

        if errors:
            print(f"[FAIL] test_class_hierarchy: {'; '.join(errors)}")
            return False

        print("[PASS] test_class_hierarchy: all classes inherit correct LiveKit base classes")
        return True
    except Exception as e:
        print(f"[FAIL] test_class_hierarchy: {e}")
        return False


if __name__ == "__main__":
    tests = [
        test_pip_metadata,
        test_imports,
        test_version,
        test_plugin_registration,
        test_instantiation,
        test_class_hierarchy,
    ]

    results = []
    for test in tests:
        results.append(test())

    print()
    passed = sum(results)
    total = len(results)
    print(f"{passed}/{total} checks passed")

    if passed < total:
        sys.exit(1)
