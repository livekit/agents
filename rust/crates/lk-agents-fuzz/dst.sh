#!/usr/bin/env bash
# The DST: the fuzz suites, then the seed campaigns over the runtime's real executors.
#
#   make -C rust dst
#
# `DST_SEEDS` sizes the campaign (32768 by default, after the `DST_SURVEY_SEEDS` 1024 of the
# survey), `DST_SOAK_SEEDS` the soak (8). Needs cargo-nextest.
set -euo pipefail
cd "$(dirname "$0")/../.."

step() { printf '\n== %s\n' "$1"; }
fail() { printf 'FAILED: %s\n' "$1" >&2; exit 1; }

gate_seeds=${DST_SEEDS:-32768}
survey_seeds=${DST_SURVEY_SEEDS:-1024}
soak_seeds=${DST_SOAK_SEEDS:-8}
for pair in "DST_SEEDS=$gate_seeds" "DST_SURVEY_SEEDS=$survey_seeds" \
    "DST_SOAK_SEEDS=$soak_seeds"; do
    [[ ${pair#*=} =~ ^[1-9][0-9]*$ ]] || fail "$pair: a campaign needs a positive number of seeds"
done

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
printf '\nDST passed\n'
