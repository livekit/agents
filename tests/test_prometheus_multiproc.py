from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit

_DEAD_PID = 2**22 + 1  # above every default pid_max


def _run(scenario: str, *args: object) -> dict[str, str]:
    # prometheus_client reads PROMETHEUS_MULTIPROC_DIR when it is imported, so
    # each scenario runs in a fresh interpreter, without the caller's value, and
    # sets the directory itself. It prints `KEY value` lines.
    env = {k: v for k, v in os.environ.items() if k.lower() != "prometheus_multiproc_dir"}
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(scenario), *map(str, args)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    return dict(line.split(" ", 1) for line in out.splitlines() if " " in line)


def test_server_run_keeps_the_metric_files_of_running_processes(tmp_path) -> None:
    out = _run(
        """
        import asyncio, os, socket, sys, time

        mp_dir, live_pid, dead_pid = sys.argv[1], sys.argv[2], sys.argv[3]
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = mp_dir

        import prometheus_client
        from prometheus_client import CollectorRegistry, generate_latest, multiprocess
        from prometheus_client.mmap_dict import MmapedDict

        from livekit.agents import AgentServer, JobContext, JobExecutorType

        for pid in (dead_pid, live_pid):
            MmapedDict(os.path.join(mp_dir, f"gauge_all_{pid}.db")).close()

        # A metric this process records before the server runs, as a warm-up step would.
        gauge = prometheus_client.Gauge("app_warmup_done", "warm-up finished")
        gauge.set(1)


        def free_port():
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                return s.getsockname()[1]


        def collect():
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            return generate_latest(registry).decode()


        async def main():
            server = AgentServer(
                job_executor_type=JobExecutorType.THREAD,
                num_idle_processes=0,
                host="127.0.0.1",
                port=free_port(),
            )

            @server.rtc_session()
            async def entrypoint(ctx: JobContext) -> None:
                pass

            run_task = asyncio.create_task(server.run(devmode=True, unregistered=True))
            # The load task writes lk_agents_child_process_count after the cleanup,
            # right after lk_agents_worker_load.
            deadline = time.monotonic() + 30
            while "lk_agents_child_process_count{" not in collect():
                if run_task.done():
                    run_task.result()
                    raise RuntimeError("AgentServer.run returned early")
                if time.monotonic() > deadline:
                    raise TimeoutError("the load task never ran")
                await asyncio.sleep(0.05)

            gauge.set(2)
            print("METRICS", collect().replace(chr(10), " | "))
            print("PID", os.getpid())
            print("FILES", " ".join(sorted(os.listdir(mp_dir))))

            await server.aclose()
            run_task.cancel()


        asyncio.run(main())
        """,
        tmp_path,
        os.getpid(),
        _DEAD_PID,
    )
    files = out["FILES"].split()

    assert f'app_warmup_done{{pid="{out["PID"]}"}} 2.0' in out["METRICS"]
    assert "lk_agents_worker_load{" in out["METRICS"]
    assert f"gauge_all_{out['PID']}.db" in files  # this process
    assert f"gauge_all_{os.getpid()}.db" in files  # a live process
    assert f"gauge_all_{_DEAD_PID}.db" not in files  # a dead process


def test_cleanup_keeps_a_metric_file_created_while_it_runs(tmp_path) -> None:
    out = _run(
        """
        import os, sys

        mp_dir = sys.argv[1]
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = mp_dir

        import prometheus_client
        from prometheus_client import CollectorRegistry, generate_latest, multiprocess

        from livekit.agents.telemetry import metrics

        real_listdir = os.listdir
        late = []


        def listdir(path):
            # Another thread records its first gauge while the cleanup runs.
            if not late:
                late.append(prometheus_client.Gauge("late_gauge", "created during cleanup"))
                late[0].set(1)
            return real_listdir(path)


        os.listdir = listdir
        metrics._clean_multiproc_dir(mp_dir)
        os.listdir = real_listdir
        late[0].set(2)

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        print("METRICS", generate_latest(registry).decode().replace(chr(10), " | "))
        print("PID", os.getpid())
        """,
        tmp_path,
    )

    assert f'late_gauge{{pid="{out["PID"]}"}} 2.0' in out["METRICS"]


def test_cleanup_removes_this_pids_files_when_this_process_writes_none(tmp_path) -> None:
    # PROMETHEUS_MULTIPROC_DIR is set after prometheus_client is imported, as
    # AgentServer(prometheus_multiproc_dir=...) does, so this process writes no
    # file and one with its pid is left over from an earlier process.
    out = _run(
        """
        import os, sys

        from prometheus_client.mmap_dict import MmapedDict

        from livekit.agents.telemetry import metrics

        mp_dir, dead_pid = sys.argv[1], sys.argv[2]
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = mp_dir
        for name in (f"counter_{os.getpid()}.db", f"gauge_all_{dead_pid}.db"):
            MmapedDict(os.path.join(mp_dir, name)).close()

        metrics._clean_multiproc_dir(mp_dir)
        print("FILES", " ".join(os.listdir(mp_dir)) or "-")
        """,
        tmp_path,
        _DEAD_PID,
    )

    assert out["FILES"] == "-"
