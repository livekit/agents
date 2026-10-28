#!/usr/bin/env bash
# Shared CI / launch-gate step. Checks never rewrite a source or baseline.
set -euo pipefail
cd "$(dirname "$0")/../.."

env PYTHONPATH="$PWD/rust/architecture" python3 -m unittest test_check test_clippy
python3 rust/architecture/check.py
cargo test --config "$PWD/rust/.cargo/config.toml" \
    --manifest-path rust/Cargo.toml --locked -q -p lk-agents-codegen --bins
cargo run --config "$PWD/rust/.cargo/config.toml" \
    --manifest-path rust/Cargo.toml --locked -q \
    -p lk-agents-codegen --bin lk-agents-codegen -- --check
