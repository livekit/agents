#!/usr/bin/env bash
# Shared CI / launch-gate step. Checks never rewrite a source or baseline.
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo_runner="${CARGO_RUNNER:-cargo}"

case "${1:-}" in
    "")
        (cd rust && cargo fmt -- --check)
        python3 -u rust/architecture/check.py
        ;;
    --tests)
        env PYTHONPATH="$PWD/rust/architecture" python3 -m unittest test_check test_clippy
        "$cargo_runner" test \
            --manifest-path rust/Cargo.toml --locked -q -p lk-agents-codegen --bins
        ;;
    *)
        echo 'Usage: check.sh [--tests]' >&2
        exit 2
        ;;
esac
