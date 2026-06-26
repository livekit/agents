# Copyright 2025 LiveKit, Inc.
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

"""Tests for the blockguard C extension.

All tests run in subprocesses to isolate static C state
and prevent deadlocks from freezing the test runner.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

TIMEOUT_SEC = 15


def _run_script(script: str, *, timeout: int = TIMEOUT_SEC) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestLifecycle:
    def test_install_uninstall(self) -> None:
        r = _run_script("""\
            import asyncio, blockguard

            async def main():
                blockguard.install(threshold_ms=5000, poll_ms=500)
                await asyncio.sleep(0.05)
                blockguard.uninstall()

            asyncio.run(main())
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "watchdog started" in r.stderr
        assert "watchdog stopped" in r.stderr

    def test_install_sleep_uninstall(self) -> None:
        r = _run_script("""\
            import asyncio, blockguard

            async def main():
                blockguard.install(threshold_ms=5000, poll_ms=100)
                await asyncio.sleep(0.05)
                blockguard.uninstall()

            asyncio.run(main())
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"


class TestDetection:
    def test_blocking_detected(self) -> None:
        r = _run_script("""\
            import asyncio, blockguard, time

            async def main():
                blockguard.install(threshold_ms=100, poll_ms=25)
                time.sleep(1.0)
                blockguard.uninstall()

            asyncio.run(main())
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "Event loop BLOCKED" in r.stderr
        assert "in main" in r.stderr

    def test_no_false_positive(self) -> None:
        r = _run_script("""\
            import asyncio, blockguard

            async def main():
                blockguard.install(threshold_ms=200, poll_ms=50)
                for _ in range(5):
                    await asyncio.sleep(0.02)
                blockguard.uninstall()

            asyncio.run(main())
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "Event loop BLOCKED" not in r.stderr

    def test_traceback_contains_file_and_line(self) -> None:
        r = _run_script("""\
            import asyncio, blockguard, time

            async def do_blocking_work():
                time.sleep(1.0)

            async def main():
                blockguard.install(threshold_ms=100, poll_ms=25)
                await do_blocking_work()
                blockguard.uninstall()

            asyncio.run(main())
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "Event loop BLOCKED" in r.stderr
        assert 'File "' in r.stderr
        assert "line" in r.stderr

    def test_blocking_without_asyncio(self) -> None:
        r = _run_script("""\
            import blockguard, time

            blockguard.install(threshold_ms=100, poll_ms=25)
            time.sleep(1.0)
            blockguard.uninstall()
            print("OK")
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "Event loop BLOCKED" in r.stderr
        assert "OK" in r.stdout

    def test_blocking_hashlib(self) -> None:
        r = _run_script("""\
            import asyncio, blockguard, hashlib

            async def main():
                blockguard.install(threshold_ms=100, poll_ms=25)
                hashlib.pbkdf2_hmac("sha256", b"password", b"salt", 5_000_000)
                blockguard.uninstall()
                print("OK")

            asyncio.run(main())
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "Event loop BLOCKED" in r.stderr
        assert "OK" in r.stdout

    def test_blocking_busy_loop(self) -> None:
        r = _run_script("""\
            import asyncio, blockguard, time

            async def main():
                blockguard.install(threshold_ms=100, poll_ms=25)
                end = time.monotonic() + 1.0
                while time.monotonic() < end:
                    pass
                blockguard.uninstall()
                print("OK")

            asyncio.run(main())
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "Event loop BLOCKED" in r.stderr
        assert "OK" in r.stdout


class TestEdgeCases:
    def test_double_install_raises(self) -> None:
        r = _run_script("""\
            import asyncio, blockguard

            async def main():
                blockguard.install(threshold_ms=5000, poll_ms=500)
                try:
                    blockguard.install(threshold_ms=5000, poll_ms=500)
                    print("ERROR: no exception raised")
                except RuntimeError as e:
                    print(f"OK: {e}")
                blockguard.uninstall()

            asyncio.run(main())
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "OK:" in r.stdout
        assert "already installed" in r.stdout

    def test_uninstall_without_install_raises(self) -> None:
        r = _run_script("""\
            import blockguard
            try:
                blockguard.uninstall()
                print("ERROR: no exception raised")
            except RuntimeError as e:
                print(f"OK: {e}")
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "OK:" in r.stdout
        assert "not installed" in r.stdout

    def test_negative_threshold_raises(self) -> None:
        r = _run_script("""\
            import blockguard
            try:
                blockguard.install(threshold_ms=-1, poll_ms=50)
                print("ERROR: no exception raised")
            except ValueError as e:
                print(f"OK: {e}")
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "OK:" in r.stdout

    def test_zero_poll_raises(self) -> None:
        r = _run_script("""\
            import blockguard
            try:
                blockguard.install(threshold_ms=100, poll_ms=0)
                print("ERROR: no exception raised")
            except ValueError as e:
                print(f"OK: {e}")
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "OK:" in r.stdout

    def test_poll_greater_than_threshold_raises(self) -> None:
        r = _run_script("""\
            import blockguard
            try:
                blockguard.install(threshold_ms=50, poll_ms=100)
                print("ERROR: no exception raised")
            except ValueError as e:
                print(f"OK: {e}")
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "OK:" in r.stdout


class TestStress:
    def test_rapid_install_uninstall_cycles(self) -> None:
        r = _run_script(
            """\
            import asyncio, blockguard

            async def main():
                for i in range(50):
                    blockguard.install(threshold_ms=500, poll_ms=50)
                    await asyncio.sleep(0.001)
                    blockguard.uninstall()
                print("OK")

            asyncio.run(main())
        """,
            timeout=30,
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "OK" in r.stdout

    def test_install_uninstall_with_blocking_load(self) -> None:
        r = _run_script(
            """\
            import asyncio, blockguard, time

            async def main():
                for i in range(10):
                    blockguard.install(threshold_ms=200, poll_ms=25)
                    time.sleep(0.05)
                    blockguard.uninstall()
                print("OK")

            asyncio.run(main())
        """,
            timeout=30,
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "OK" in r.stdout

    def test_uninstall_during_active_detection(self) -> None:
        r = _run_script(
            """\
            import asyncio, blockguard, time

            async def main():
                blockguard.install(threshold_ms=50, poll_ms=10)
                time.sleep(0.2)
                blockguard.uninstall()
                print("OK")

            asyncio.run(main())
        """,
            timeout=15,
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "OK" in r.stdout

    def test_many_short_blocks(self) -> None:
        r = _run_script("""\
            import asyncio, blockguard, time

            async def main():
                blockguard.install(threshold_ms=500, poll_ms=50)
                for _ in range(20):
                    time.sleep(0.005)
                    await asyncio.sleep(0.01)
                blockguard.uninstall()
                print("OK")

            asyncio.run(main())
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "Event loop BLOCKED" not in r.stderr
        assert "OK" in r.stdout

    def test_reinstall_after_uninstall(self) -> None:
        r = _run_script("""\
            import asyncio, blockguard, time

            async def main():
                blockguard.install(threshold_ms=100, poll_ms=25)
                await asyncio.sleep(0.05)
                blockguard.uninstall()

                blockguard.install(threshold_ms=100, poll_ms=25)
                time.sleep(1.0)
                blockguard.uninstall()
                print("OK")

            asyncio.run(main())
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "Event loop BLOCKED" in r.stderr
        assert "OK" in r.stdout


class TestPythonWrapper:
    def test_wrapper_import(self) -> None:
        r = _run_script("""\
            from livekit import blockguard
            print(f"version={blockguard.__version__}")
            print("OK")
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "OK" in r.stdout
        assert "version=" in r.stdout

    def test_wrapper_lifecycle(self) -> None:
        r = _run_script("""\
            import asyncio
            from livekit import blockguard

            async def main():
                blockguard.install(threshold_ms=5000, poll_ms=500)
                await asyncio.sleep(0.05)
                blockguard.uninstall()
                print("OK")

            asyncio.run(main())
        """)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "OK" in r.stdout
