#!/usr/bin/env bash
# The voice core's launch gate: every suite the DST has, in one command, failing on any finding
# outside the known ones (`check::KNOWN_FINDINGS`) and on anything that did not run. COVERAGE.md,
# section 12, says what each step proves.
#
#   rust/crates/lk-agents-fuzz/launch-gate.sh
#
# `VOICE_GATE_SEEDS` sizes the large campaign (4096 by default, after the 1024 of the survey).
# Verdicts depend on no clock but the simulation's; `VOICE_HANG_S` is the watchdog that ends a
# run blocked for real (600 s by default).
set -euo pipefail
cd "$(dirname "$0")/../.."

step() { printf '\n== %s\n' "$1"; }

# A campaign is one ignored test: it must run, pass, and say what it reached.
campaign() {
    local name=$1 test=$2
    shift 2
    step "$name"
    local out
    if ! out=$(env "$@" cargo test -q --release -p lk-agents-fuzz --test "$test" -- --ignored \
        --nocapture 2>&1); then
        grep -E "^  NEW " <<<"$out" || true
        printf '%s\n' "$out" | tail -40
        echo "FAILED: $name"
        exit 1
    fi
    if ! grep -q "test result: ok. 1 passed" <<<"$out"; then
        printf '%s\n' "$out" | tail -20
        echo "FAILED: $name ran no test"
        exit 1
    fi
    grep -E "^[0-9]+ seeds from|^reached |calls reached their close|EXCLUDED|known findings this" \
        <<<"$out" || true
}

step "build"
cargo check -q --workspace --all-targets

step "voice core unit tests"
cargo test -q -p lk-agents-voice

step "fuzz suites: contract registry, probes, replay corpus, determinism, regressions"
cargo test -q --release -p lk-agents-fuzz

campaign "survey, 1024 seeds" survey VOICE_SEEDS=1024
campaign "campaign, ${VOICE_GATE_SEEDS:-4096} more seeds" survey \
    VOICE_FIRST_SEED=1024 VOICE_SEEDS="${VOICE_GATE_SEEDS:-4096}"
campaign "soak" soak

printf '\nlaunch gate passed\n'
