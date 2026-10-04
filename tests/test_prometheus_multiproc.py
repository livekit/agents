from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit

# prometheus_client reads PROMETHEUS_MULTIPROC_DIR when it is imported, so the
# scenario runs in a fresh interpreter.
_SCENARIO = textwrap.dedent(
    """
    import asyncio, os, socket, sys

    mp_dir, live_pid = sys.argv[1], int(sys.argv[2])
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = mp_dir

    import prometheus_client
    from prometheus_client import CollectorRegistry, generate_latest, multiprocess
    from prometheus_client.mmap_dict import MmapedDict

    from livekit.agents import AgentServer, JobContext, JobExecutorType

    own_pid = os.getpid()
    dead_pid = 2**22 + 1  # above every default pid_max

    # Files of earlier processes: a dead one, one that had this pid, and a live one.
    for name in (f"gauge_all_{dead_pid}.db", f"counter_{own_pid}.db", f"gauge_all_{live_pid}.db"):
        MmapedDict(os.path.join(mp_dir, name)).close()

    # A metric this process records before the server runs, as a warm-up step would.
    gauge = prometheus_client.Gauge("app_warmup_done", "warm-up finished")
    gauge.set(1)


    def free_port():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


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
        await asyncio.sleep(1.5)  # a few load updates
        gauge.set(2)

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        print(generate_latest(registry).decode())
        print("PID", own_pid)
        print("FILES", " ".join(sorted(os.listdir(mp_dir))))

        await server.aclose()
        run_task.cancel()


    asyncio.run(main())
    """
)


def test_server_run_keeps_the_metric_files_of_running_processes(tmp_path) -> None:
    out = subprocess.run(
        [sys.executable, "-c", _SCENARIO, str(tmp_path), str(os.getpid())],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    lines = out.splitlines()
    server_pid = next(line.split()[1] for line in lines if line.startswith("PID "))
    files = set(next(line.split()[1:] for line in lines if line.startswith("FILES ")))

    assert f'app_warmup_done{{pid="{server_pid}"}} 2.0' in out
    assert "lk_agents_worker_load{" in out
    assert f"gauge_all_{server_pid}.db" in files  # held open by the server process
    assert f"gauge_all_{os.getpid()}.db" in files  # a live process
    assert f"gauge_all_{2**22 + 1}.db" not in files  # a dead process
    assert f"counter_{server_pid}.db" not in files  # this pid, not held open
