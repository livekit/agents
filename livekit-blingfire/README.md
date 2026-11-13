# LiveKit BlingFire

This source build supplies the four existing `livekit.blingfire` tokenizer operations for standard, GIL-enabled CPython 3.15.2 on macOS arm64 and glibc Linux x86_64/aarch64. It builds Microsoft's public engine and the Python boundary from source; it contains no CPython 3.14 binary payload.

## Build

Run `uv build --no-sources --python 3.15.2 livekit-blingfire` from the repository root. The PEP 517 backend verifies and extracts the included source archive, then builds the engine and extension with one compiler worker. Build the generated source distribution with the same command to verify the source release is self-contained. Linux release wheels require `auditwheel repair --plat manylinux_2_28_ARCH`; the native wheel workflow owns that repair.

## Verify

Install the wheel and `tests/requirements.txt` into a clean CPython 3.15.2 environment, then run `python tests/qualify.py outcomes.json` from this folder. The qualifier records the imported native producer and requires exactly 3,372 outcomes from 843 frozen cases, including Unicode, embedded NULs, offsets, long text and errors. Expected outcomes were captured from the published 1.1.0 wheels on macOS with CPython 3.14.8 and Linux with CPython 3.14.10; Linux's four empty-input errors differ from macOS and remain preserved.

## Source

`vendor/manifest.json` records every selected file from Microsoft commit `18e9a19e586095fb60d629fa850fb610d5bca605`, its Git blob and SHA256. The included archive is a 539-file source projection, not the complete upstream archive or unpublished LiveKit native source. The engine and its word/sentence models are unchanged. Apache-2.0, MIT and BSD notices remain in the package.

This package addresses the BlingFire part of [#8237](https://github.com/livekit/agents/issues/8237). `livekit-local-inference` still needs genuine CPython 3.15 artifacts or source; this change does not claim full Agents support or change the EOT model. Other Python versions, free-threaded builds and subinterpreters are outside this source build's qualification.
