# Native Host Probes

We exercise the production facade and process protocol with checked working fakes. These probes require Python 3.15.2, NumPy and the built `_napi_host315` extension; they do not execute the official engine.

Build the source-owned fake into a fresh directory before the parent-loss probe. Linux shared-library builds retain unresolved Node-API symbols; on macOS, add `-Wl,-undefined,dynamic_lookup` to this command:

```sh
mkdir .fake-native
c++ -std=c++20 -shared -fPIC -Wall -Wextra -Werror -I native/include native/fake.cpp -o .fake-native/source-owned-fake.node
```

Run the public facade probes from the package root in an isolated environment with a finite outer deadline:

```sh
.host-venv/bin/python -I -B tests/probe_boundary.py
.host-venv/bin/python -I -B tests/probe_transport.py
LIVEKIT_BOUNDARY_NATIVE_DIR="$PWD/.fake-native" .host-venv/bin/python -I -B tests/probe_lifecycle.py
```

Set `LIVEKIT_BOUNDARY_PACKAGE_ROOT` to the source root only when testing a source projection rather than an installed package. Run each probe in a separate process because module shutdown intentionally closes admission permanently.

The boundary probe checks concurrent EOT/VAD calls, recurrent affinity, reset isolation, array layout compatibility and worker retirement. The transport and lifecycle probes add partial-spawn failures, partial replies, deadlines, close during work and parent loss.

The [C++ fake](../native/fake.cpp) can replace the Python fake backend through `LIVEKIT_BOUNDARY_FAKE_ADDON`. Build it as a shared library against the frozen headers, retaining unresolved Node-API symbols for the loaded host to supply.

The separate `probe_sdk_cleanup.py` runs checked native close subclasses through actual SDK cleanup methods. It requires the SDK cleanup patch, an installed source path and an output path; it does not open model weights or mutate installed engine metadata.

None of these results establishes genuine model quality. Keep real-engine parity, SDK workflows, per-platform packaging and paired latency evidence separate.

`probe_restart.py` checks explicit startup, closed admission and retry after failed retirement. `probe_fork.py` holds parent locks across a real fork, checks copied descriptors and reused descriptor numbers, and verifies that foreign lifecycle calls and finalizers leave the parent workers alone.
