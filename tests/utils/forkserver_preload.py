from __future__ import annotations

import gc
import json
import multiprocessing as mp
import weakref
from multiprocessing.connection import Connection


class _Cycle:
    reference: _Cycle


def _inspect_gc(connection: Connection) -> None:
    freeze_count = gc.get_freeze_count()

    cycle = _Cycle()
    cycle.reference = cycle
    cycle_reference = weakref.ref(cycle)
    del cycle
    gc.collect()

    connection.send(
        {
            "freeze_count": freeze_count,
            "child_cycle_collected": cycle_reference() is None,
        }
    )
    connection.close()


if __name__ == "__main__":
    context = mp.get_context("forkserver")
    context.set_forkserver_preload(["livekit.agents.ipc._preload_freeze"])
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=_inspect_gc, args=(child_connection,))

    process.start()
    child_connection.close()
    outcome = parent_connection.recv()
    parent_connection.close()
    process.join()

    if process.exitcode != 0:
        raise SystemExit(process.exitcode)

    print(json.dumps(outcome))
