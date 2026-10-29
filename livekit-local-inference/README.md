# LiveKit Local Inference for CPython 3.15

We expose LiveKit's existing 0.2.7 native engine through a source-built CPython 3.15 host. The model, weights and recurrent state stay in the official engine; the application and every Python worker use 3.15.2.

- [Build](#build)
- [Use With Agents](#use-with-agents)
- [Ownership](#ownership)
- [Tests](#tests)
- [Engine And Headers](#engine-and-headers)
- [License And Support](#license-and-support)

## Build

Build a platform projection from the official npm archives listed in [engine-artifacts.json](engine-artifacts.json). Download those exact archives without installing packages or running npm scripts; the preparer verifies archive, addon and notice hashes before copying one addon into a fresh source directory.

Use an isolated Python 3.15.2 environment to build the prepared source:

```sh
uv venv --python 3.15.2 .host-venv
uv --no-config pip install --python .host-venv/bin/python --require-hashes -r build-requirements.txt
.host-venv/bin/python prepare.py --archives ARCHIVE_DIRECTORY --platform darwin-arm64 NEW_SOURCE_DIRECTORY
uv --no-config build --wheel --python .host-venv/bin/python --no-build-isolation --out-dir EMPTY_WHEEL_DIRECTORY NEW_SOURCE_DIRECTORY
```

The other projections are `linux-arm64-gnu` and `linux-x64-gnu`. Use one compiler worker. Inspect every ELF payload before repairing Linux wheels; the final tag must cover both the host and addon requirements. The engine alone requires glibc 2.34.

The source tree builds the host and Python facade. It retains the official engine as a fixed binary input; it does not rebuild that engine from source, download models during import or require a Node executable at runtime.

## Use With Agents

Install the selected wheel into a qualified LiveKit Agents Python 3.15.2 environment. Its `livekit.local_inference` namespace keeps the SDK's `EOT`, `VAD`, `init_eot` and `init_vad` interfaces.

VAD accepts one 512-sample int16 window and owns recurrent state for each stream. EOT accepts up to 19,200 int16 samples; the SDK handles audio framing and model selection.

This host package does not establish Python 3.15 support for every Agents dependency. The SDK metadata, RTC, tokenizer and other native packages need their own qualified distributions.

## Ownership

Call `start()` when opening an application lifetime and join `shutdown()` before it ends. Shutdown rejects late warmup and inference; startup reopens admission only after every previous worker has retired, and failed retirement must be retried.

Each model has explicit `close()` ownership and supports a context manager. VAD state stays with its worker generation; lost state requires a fresh VAD instance.

EOT prewarming retains one worker. A second starts on parallel demand and pays its model load cost once; closing the last EOT model retires that second worker and preserves only an explicitly warmed primary.

The backend declares `FORK_SAFE_PREWARM = False`, so the SDK skips automatic import-time model warmup. A job creates its own pool; a forked child closes copied IPC handles and rejects inherited model operations before taking their locks.

SDK shutdown awaits physical worker retirement. A native pipe watcher also retires its own child when the parent disappears, including when foreign code holds the GIL.

## Tests

The [working-fake probes](tests/README.md) exercise framing, deadlines, recurrent-state isolation, partial startup, restart, fork detachment, close and parent loss without executing the vendor model. The C++ fake uses the host's actual Node-API callbacks and ownership boundary.

Model parity and latency need separate checks against the genuine engine on each platform. Fake success, a successful build and ABI symbol coverage do not establish model quality or a performance improvement.

The native source workflow builds twice on macOS ARM64 and Linux ARM64/x64, then runs the installed facade and fake cohorts. It does not execute the vendor model or publish release wheels.

The prepared engine controls the host architecture and wheel platform. A universal2 macOS interpreter builds an ARM64 host for the ARM64 engine; the wheel keeps its compiled minimum-OS check.

## Engine And Headers

The artifact index pins the official `@livekit/local-inference` 0.2.7 packages and records the source commit from their published metadata. A prepared source directory records its own input hashes in `SOURCE_MANIFEST.json`.

The host implements the [Node-API](https://nodejs.org/api/n-api.html) operations used by that engine. Frozen Node.js 26.12.1 declarations and their complete notices live in [native](native/README.md); the host does not link libnode, V8 or libuv.

## License And Support

Keep the Apache-2.0, Node MIT and [LiveKit Model License](MODEL_LICENSE) notices with each distribution. The model license limits model use to LiveKit Agents.

Use the [Agents issue tracker](https://github.com/livekit/agents/issues) for reproducible bugs and LiveKit's [security policy](https://livekit.com/security/policy) for security reports. For turn-detector behavior and hosted alternatives, see the [LiveKit documentation](https://docs.livekit.io/agents/logic/turns/turn-detector/).

We can remove this adapter when an upstream Python distribution supports the required interpreter and platforms and passes the same model, SDK, lifecycle, artifact and latency checks.
