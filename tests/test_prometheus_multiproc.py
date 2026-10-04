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
        import asyncio, os, sys, threading, time

        mp_dir, live_pid, dead_pid = sys.argv[1], sys.argv[2], sys.argv[3]
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = mp_dir

        import prometheus_client
        from prometheus_client import CollectorRegistry, generate_latest, multiprocess
        from prometheus_client.mmap_dict import MmapedDict

        from livekit.agents import AgentServer, JobContext, JobExecutorType
        from livekit.agents.telemetry import metrics as lk_metrics

        for pid in (dead_pid, live_pid):
            MmapedDict(os.path.join(mp_dir, f"gauge_all_{pid}.db")).close()

        # A metric this process records before the server runs, as a warm-up step would.
        gauge = prometheus_client.Gauge("app_warmup_done", "warm-up finished")
        gauge.set(1)


        def collect():
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            return generate_latest(registry).decode()


        async def main():
            server = AgentServer(
                job_executor_type=JobExecutorType.THREAD,
                num_idle_processes=0,
                host="127.0.0.1",
                port=0,
            )

            @server.rtc_session()
            async def entrypoint(ctx: JobContext) -> None:
                pass

            # The load task writes lk_agents_child_process_count after the cleanup,
            # right after lk_agents_worker_load. Wait until that write returns, so
            # the collector never reads a file that is still being initialized.
            child_count_written = threading.Event()
            update_child_proc_count = lk_metrics._update_child_proc_count

            def update_and_signal():
                update_child_proc_count()
                child_count_written.set()

            lk_metrics._update_child_proc_count = update_and_signal

            run_task = asyncio.create_task(server.run(devmode=True, unregistered=True))
            deadline = time.monotonic() + 30
            while not child_count_written.is_set():
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


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd to list open files")
def test_cleanup_removes_this_pids_files_from_a_directory_it_does_not_write_to(
    tmp_path,
) -> None:
    # Multiprocess mode is on with directory A, and the server is given directory
    # B, which holds a file with this pid from an earlier process.
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    out = _run(
        """
        import os, sys

        a_dir, b_dir = sys.argv[1], sys.argv[2]
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = a_dir

        from prometheus_client import Gauge
        from prometheus_client.mmap_dict import MmapedDict

        from livekit.agents.telemetry import metrics

        Gauge("app_warmup_done", "", multiprocess_mode="all").set(1)
        MmapedDict(os.path.join(b_dir, f"gauge_all_{os.getpid()}.db")).close()

        metrics._clean_multiproc_dir(b_dir)
        print("A", " ".join(sorted(os.listdir(a_dir))) or "-")
        print("B", " ".join(os.listdir(b_dir)) or "-")
        """,
        a_dir,
        b_dir,
    )

    assert out["B"] == "-"
    assert out["A"] != "-"


def test_cleanup_keeps_this_pids_open_files_after_a_directory_switch(tmp_path) -> None:
    # A gauge is recorded in directory A, PROMETHEUS_MULTIPROC_DIR moves to B,
    # and A is cleaned again, as a server that returns to A would.
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    out = _run(
        """
        import os, sys

        a_dir, b_dir = sys.argv[1], sys.argv[2]
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = a_dir

        from prometheus_client import CollectorRegistry, Gauge, generate_latest, multiprocess

        from livekit.agents.telemetry import metrics

        gauge = Gauge("app_warmup_done", "", multiprocess_mode="all")
        gauge.set(1)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = b_dir

        metrics._clean_multiproc_dir(a_dir)
        gauge.set(2)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry, path=a_dir)
        print("METRICS", generate_latest(registry).decode().replace(chr(10), " | "))
        print("PID", os.getpid())
        """,
        a_dir,
        b_dir,
    )

    assert f'app_warmup_done{{pid="{out["PID"]}"}} 2.0' in out["METRICS"]


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd to list open files")
def test_cleanup_in_the_current_directory_removes_only_this_pids_stale_files(tmp_path) -> None:
    # Multiprocess mode is on with this directory. counter_<pid>.db is left over
    # from an earlier process with this pid. gauge_max_<pid>.db is created but not
    # sized yet, as a file is while another thread opens it.
    out = _run(
        """
        import os, sys

        mp_dir = sys.argv[1]
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = mp_dir

        from prometheus_client import Gauge
        from prometheus_client.mmap_dict import MmapedDict

        from livekit.agents.telemetry import metrics

        Gauge("app_warmup_done", "", multiprocess_mode="all").set(1)
        MmapedDict(os.path.join(mp_dir, f"counter_{os.getpid()}.db")).close()
        open(os.path.join(mp_dir, f"gauge_max_{os.getpid()}.db"), "ab").close()

        metrics._clean_multiproc_dir(mp_dir)
        print("PID", os.getpid())
        print("FILES", " ".join(sorted(os.listdir(mp_dir))))
        """,
        tmp_path,
    )

    pid = out["PID"]
    assert out["FILES"].split() == [f"gauge_all_{pid}.db", f"gauge_max_{pid}.db"]
