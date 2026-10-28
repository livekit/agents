#!/usr/bin/env bash
# The known-bug benchmark: how fast each queue policy finds bugs the DST found and we fixed.
#
#   rust/crates/lk-agents-fuzz/bench/known_bugs.sh [bug ...]
#
# Each bugs/<name>.patch is a fix commit's product diff. For each, the fix is reverted in the
# working tree (never committed), `tests/pos.rs bench` runs every policy over the same seeds and
# then over one failing case's schedules, and the fix is put back. A bug is detected by the
# contracts that fail with it and never on the tree as it stands (the baseline run). A patch the
# tree has moved past is reported stale, not skipped silently.
#
# `VOICE_SEEDS` (16384) campaign seeds, `VOICE_SETUPS` (1024) conversations on a failing setup and
# `VOICE_SCHEDULES` (512) schedules of a failing case, per policy;
# `VOICE_BENCH_OUT` keeps each run's output (a temporary directory by default). Needs a working
# tree with no changes under the crates the patches touch.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../../.."
root="$(git rev-parse --show-toplevel)"
out="${VOICE_BENCH_OUT:-$(mktemp -d)}"
mkdir -p "$out"
export VOICE_SEEDS="${VOICE_SEEDS:-16384}" VOICE_SCHEDULES="${VOICE_SCHEDULES:-512}"

bench() {
    cargo test -q --release -p lk-agents-fuzz --test pos -- --ignored --nocapture --exact bench \
        > "$out/$1.txt" 2>&1
}

if [ $# -eq 0 ]; then
    set -- $(cd "$here/bugs" && ls ./*.patch | sed 's|^\./||; s|\.patch$||')
fi
if [ ! -s "$out/baseline.txt" ]; then
    echo "== baseline"
    bench baseline
fi
for bug in "$@"; do
    patch="$here/bugs/$bug.patch"
    if ! git -C "$root" apply -R --check "$patch" 2>/dev/null; then
        echo "== $bug: STALE (the fix no longer reverts cleanly)"
        echo "STALE" > "$out/$bug.txt"
        continue
    fi
    echo "== $bug"
    git -C "$root" apply -R "$patch"
    trap 'git -C "$root" apply "$patch"' EXIT
    bench "$bug" || echo "== $bug: the run failed (see $out/$bug.txt)"
    git -C "$root" apply "$patch"
    trap - EXIT
done


# The table: per bug, the contracts it fails that the baseline never does, at three levels: the
# campaign (fresh seeds), the setup (its first failing case's setup, fresh conversations) and the
# program (that case's inputs, fresh schedules). Each cell: hits / tries, then the median over the
# windows of the tries to the first hit.
echo
awk -F'\t' -v bugs="$*" '
    FNR == 1 { bug = FILENAME; sub(/.*\//, "", bug); sub(/\.txt$/, "", bug) }
    $1 != "TSV" { next }
    bug == "baseline" && $2 != "campaign" && $2 != "finding" && $5 > 0 { base[$4] = 1 }
    bug == "baseline" && $2 == "campaign" { cost[$3] = $4; edges[$3] = $6; found[$3] = $7 }
    bug != "baseline" && ($2 == "seeds" || $2 == "setup" || $2 == "program") {
        cell[bug, $4, $2, $3] = $5 "/" $6 " " $7
        hits[bug, $4, $2, $3] = $5
        tries[$2] = $6
        if ($2 == "program") contracts[bug, $4] = 1
    }
    END {
        split("Arrival BasicPos FullPos TaskPos", policies, " ")
        split("seeds setup program", levels, " ")
        printf "%-26s %-34s %-8s", "bug", "contract", "level"
        for (p = 1; p <= 4; p++) printf " %-16s", policies[p]
        printf "\n"
        n = split(bugs, list, " ")
        for (b = 1; b <= n; b++) {
            detected = 0
            for (key in contracts) {
                split(key, parts, SUBSEP)
                if (parts[1] != list[b] || (parts[2] in base)) continue
                detected = 1; rows++
                for (l = 1; l <= 3; l++) {
                    printf "%-26s %-34s %-8s", list[b], parts[2], levels[l]
                    for (p = 1; p <= 4; p++) {
                        printf " %-16s", cell[list[b], parts[2], levels[l], policies[p]]
                        r = hits[list[b], parts[2], levels[l], policies[p]] / tries[levels[l]]
                        lg[l, p] += log(r > 0 ? r : 0.5 / tries[levels[l]])
                    }
                    printf "\n"
                }
            }
            if (!detected) printf "%-26s %s\n", list[b], "not detected by any policy"
        }
        if (rows) {
            printf "\ngeometric mean hit ratio over %d detected contracts (a zero counted as half a hit):\n", rows
            for (l = 1; l <= 3; l++) {
                printf "  %-8s", levels[l]
                for (p = 1; p <= 4; p++) printf " %s %.5f", policies[p], exp(lg[l, p] / rows)
                printf "\n"
            }
        }
        printf "\nbaseline campaign: seeds/s, edges, distinct findings\n"
        for (p = 1; p <= 4; p++)
            printf "  %-9s %8s %6s %4s\n", policies[p], cost[policies[p]], edges[policies[p]],
                found[policies[p]]
    }
' "$out/baseline.txt" $(for bug in "$@"; do echo "$out/$bug.txt"; done)
echo "outputs: $out"
