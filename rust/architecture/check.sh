#!/usr/bin/env bash
# Shared CI / launch-gate step. Checks never rewrite a source or baseline.
set -euo pipefail
cd "$(dirname "$0")/../.."

run() {
    local label=$1
    shift
    local started=$SECONDS rc=0
    "$@" || rc=$?
    local line
    line="$(date +%T) s0-architecture $label rc=$rc $((SECONDS-started))s"
    mkdir -p "${ARCHITECTURE_TIMINGS_DIR:-/tmp/plan-align}"
    echo "$line" | tee -a "${ARCHITECTURE_TIMINGS_DIR:-/tmp/plan-align}/timings.log"
    return "$rc"
}

run counterexamples env PYTHONPATH="$PWD/rust/architecture" python3 -m unittest test_check test_clippy
run dependencies-lengths-baseline-history python3 rust/architecture/check.py
run clippy python3 rust/architecture/clippy.py
run export-counterexamples cargo test --config "$PWD/rust/.cargo/config.toml" \
    --manifest-path rust/Cargo.toml --locked -q \
    -p lk-agents-codegen --bin lk-agents-codegen placement
run codegen cargo run --config "$PWD/rust/.cargo/config.toml" \
    --manifest-path rust/Cargo.toml --locked -q \
    -p lk-agents-codegen --bin lk-agents-codegen -- --check
