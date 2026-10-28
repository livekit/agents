#!/usr/bin/env bash
# The launch gate: every suite the merge gate runs, in one command. Fails on any failed step, on a
# finding outside the known ones (`check::KNOWN_FINDINGS`), on zero seeds and on a workspace crate
# the coverage table below does not place. COVERAGE.md, section 12, says what each step proves.
#
#   rust/crates/lk-agents-fuzz/launch-gate.sh
#
# `VOICE_GATE_SEEDS` sizes the large campaign (32768 by default, after the `VOICE_SURVEY_SEEDS`
# 1024 of the survey), `VOICE_SOAK_SEEDS` the soak (8). `VOICE_FULL_WINDOW=1` (nightly) makes the
# large campaign the full window: a known finding it no longer reaches whose fixing regression
# exists fails it until it comes off `known_findings.tsv`. Verdicts depend on no clock but the
# simulation's; `VOICE_HANG_S` is the watchdog that ends a run blocked for real (600 s by default).
# Every verdict is a test runner's exit status: campaigns run under nextest, which also fails when
# its selection runs no test. Needs cargo-nextest and the repository's venv (`make install`), or
# `PYO3_PYTHON` naming a Python.
set -euo pipefail
cd "$(dirname "$0")/../.."

step() { printf '\n== %s\n' "$1"; }
fail() { printf 'FAILED: %s\n' "$1" >&2; exit 1; }

full_window=${VOICE_FULL_WINDOW:-}
unset VOICE_FULL_WINDOW
gate_seeds=${VOICE_GATE_SEEDS:-32768}
survey_seeds=${VOICE_SURVEY_SEEDS:-1024}
soak_seeds=${VOICE_SOAK_SEEDS:-8}
for pair in "VOICE_GATE_SEEDS=$gate_seeds" "VOICE_SURVEY_SEEDS=$survey_seeds" \
    "VOICE_SOAK_SEEDS=$soak_seeds"; do
    [[ ${pair#*=} =~ ^[1-9][0-9]*$ ]] || fail "$pair: a campaign needs a positive number of seeds"
done

# What the gate does with each workspace crate. Executed: its tests run here. Compiled: built
# with every target (`cargo check --workspace --all-targets`), its tests not run here. Outside:
# verified by another required job, named. A crate in no list, or in two, fails the gate.
executed=(lk-agents-abi lk-agents-fuzz lk-agents-inference lk-agents-voice)
compiled=(
    "lk-agents-bindgen: no tests of its own; exercised through lk-agents-py's generation"
    "lk-agents-host-derive: proc macros; exercised by every crate that derives"
    "lk-agents-media: tests not in the gate yet (K-12 closure)"
    "lk-agents-rtc: tests not in the gate yet (K-12 closure)"
    "lk-agents-worker: tests not in the gate yet; worker DST is K-2"
    "lk-agents-py: binding tests not in the gate yet (B-19 conformance, K-12 closure)"
    "lk-agents-codegen: unit tests not in the gate yet; codegen --check runs in ci.yml core"
    "lk-agents-runtime: tests not in the gate yet (runtime verification is K-30)"
)
outside=(
    "lk-agents-storage: ci.yml storage-tests (cargo test); storage DST is K-3"
    "lk-agents-a2a: ci.yml storage-tests (cargo test); A2A DST is K-4"
)

step "coverage"
crates=$(cargo metadata --no-deps --format-version 1 \
    | python3 -c 'import json, sys; print("\n".join(sorted(p["name"] for p in json.load(sys.stdin)["packages"])))')
placed=$(
    printf '%s\n' "${executed[@]}"
    printf '%s\n' "${compiled[@]}" "${outside[@]}" | cut -d: -f1
)
printf '  %-24s %s\n' "crate" "gate"
for crate in "${executed[@]}"; do printf '  %-24s executed\n' "$crate"; done
for row in "${compiled[@]}"; do printf '  %-24s compiled only:%s\n' "${row%%:*}" "${row#*:}"; done
for row in "${outside[@]}"; do printf '  %-24s outside:%s\n' "${row%%:*}" "${row#*:}"; done
printf '  not in the gate yet: pytest --unit --audio_eot (tests.yml), B-19 conformance, I-27\n'
printf '  backend contracts, G-12 IPC stress, K-30 runtime verification (K-12 closure)\n'
unplaced=$(comm -23 <(echo "$crates") <(echo "$placed" | sort))
twice=$(echo "$placed" | sort | uniq -d)
stale=$(comm -13 <(echo "$crates") <(echo "$placed" | sort -u))
[ -z "$unplaced" ] || fail "workspace crates the gate does not place: $unplaced"
[ -z "$twice" ] || fail "crates placed twice: $twice"
[ -z "$stale" ] || fail "placed crates that are not in the workspace: $stale"

step "fakes' calibration matches its recordings"
python3 crates/lk-agents-fuzz/corpus/derive.py --check

step "build"
cargo check -q --workspace --all-targets

step "voice core unit tests"
cargo test -q -p lk-agents-voice

# The provider clients the DST does not run: the gateway's TTS, STT and LLM, the host bridges.
step "provider clients' unit tests"
cargo test -q -p lk-agents-inference -p lk-agents-abi

# The abi tests behind the binding's `python` feature (the host bridge, the value crossings, the
# fork guard) against a live interpreter: the repository's venv unless `PYO3_PYTHON` names one. A
# process per test (nextest), since some set process globals (the fork guard, cached host classes).
step "abi tests against the interpreter"
py=${PYO3_PYTHON:-$PWD/../.venv/bin/python}
PYO3_PYTHON=$py PYTHONHOME=$("$py" -c 'import sys; print(sys.base_prefix)') \
    cargo nextest run --cargo-quiet --status-level fail --final-status-level fail \
    -p lk-agents-abi --features python

step "fuzz suites: scheduler, contract registry, probes, determinism, regressions"
cargo test -q --release -p lk-agents-fuzz

# A campaign is one ignored test; nextest fails when the selection runs nothing.
campaign() {
    local name=$1 test=$2 function=$3
    shift 3
    step "$name"
    env "$@" cargo nextest run --cargo-quiet --release -p lk-agents-fuzz --test "$test" \
        --run-ignored only --no-capture --status-level fail --final-status-level fail \
        -E "test(=$function)"
}

campaign "survey, $survey_seeds seeds" survey survey VOICE_SEEDS="$survey_seeds"
campaign "campaign, $gate_seeds more seeds" survey survey \
    VOICE_FIRST_SEED="$survey_seeds" VOICE_SEEDS="$gate_seeds" ${full_window:+VOICE_FULL_WINDOW=1}
campaign "soak, $soak_seeds calls" soak nothing_grows_with_the_conversation \
    VOICE_SOAK_SEEDS="$soak_seeds"

printf '\nlaunch gate passed\n'
