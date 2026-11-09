#!/usr/bin/env bash
# The launch gate: every Rust test suite once, the Python worker journeys, and the DST campaigns.
# The Rust rules run separately (`make -C rust check`).
#
#   rust/crates/lk-agents-fuzz/launch-gate.sh
#
# `VOICE_GATE_SEEDS` sizes the campaign (32768 by default, after the `VOICE_SURVEY_SEEDS` 1024 of
# the survey), `VOICE_SOAK_SEEDS` the soak (8). Needs cargo-nextest and the repository's venv
# (`make install`), or `PYO3_PYTHON` naming a Python.
set -euo pipefail
cd "$(dirname "$0")/../.."

step() { printf '\n== %s\n' "$1"; }
fail() { printf 'FAILED: %s\n' "$1" >&2; exit 1; }

gate_seeds=${VOICE_GATE_SEEDS:-32768}
survey_seeds=${VOICE_SURVEY_SEEDS:-1024}
soak_seeds=${VOICE_SOAK_SEEDS:-8}
for pair in "VOICE_GATE_SEEDS=$gate_seeds" "VOICE_SURVEY_SEEDS=$survey_seeds" \
    "VOICE_SOAK_SEEDS=$soak_seeds"; do
    [[ ${pair#*=} =~ ^[1-9][0-9]*$ ]] || fail "$pair: a campaign needs a positive number of seeds"
done

py=${PYO3_PYTHON:-$PWD/../.venv/bin/python}
agent_python_path=${PYTHONPATH:-$PWD/../livekit-agents-next}
python_home=$("$py" -c 'import sys; print(sys.base_prefix)')
site_packages=$("$py" -c 'import site; print(site.getsitepackages()[0])')

step "Rust tests"
cargo test -q --workspace --exclude lk-agents-py --exclude lk-agents-fuzz

step "Python binding contracts"
PYO3_PYTHON=$py PYTHONHOME=$python_home cargo test -q -p lk-agents-py --features ffmpeg
# The abi tests behind `python` set process globals (fork guard, cached host classes): nextest
# gives each its own process.
PYO3_PYTHON=$py PYTHONHOME=$python_home PYTHONPATH=$site_packages \
    cargo nextest run --cargo-quiet --status-level fail --final-status-level fail \
    -p lk-agents-abi --features python

step "worker generated API, CLI and endpoint contracts"
PYTHONPATH="$agent_python_path:$PWD/.." "$py" -m pytest \
    ../tests/conformance/test_worker_records.py ../tests/conformance/test_python_bootstrap.py \
    ../tests/test_endpoint_routes.py --unit -q

step "bounded real IPC, preload and shutdown journeys"
LIVEKIT_IPC_SOAK_BATCHES=3 LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE=0 \
    PYTHONPATH="$agent_python_path:$PWD/.." "$py" -m pytest \
    ../tests/conformance/test_worker_ipc_resources.py \
    ../tests/conformance/test_worker_preload.py ../tests/conformance/test_cli_shutdown.py \
    ../tests/test_it_ipc.py --e2e -q

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
    VOICE_FIRST_SEED="$survey_seeds" VOICE_SEEDS="$gate_seeds"
campaign "soak, $soak_seeds calls" soak nothing_grows_with_the_conversation \
    VOICE_SOAK_SEEDS="$soak_seeds"
printf '\nlaunch gate passed\n'
