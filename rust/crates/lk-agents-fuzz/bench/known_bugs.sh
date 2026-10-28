#!/usr/bin/env bash
# The known-bug benchmark: how fast each queue policy finds bugs the DST found and we fixed.
#
#   rust/crates/lk-agents-fuzz/bench/known_bugs.sh [--rev <commit>] [--compare <summary.tsv>] [bug ...]
#
# Each bugs/<name>.patch is a fix commit's product diff. The benchmark runs on a copy of the
# rust/ tree, never on the checkout: the tree as it stands (HEAD plus uncommitted and untracked
# files), or `--rev`'s commit. For each bug the fix is reverted in the copy, `tests/pos.rs bench`
# runs every policy over the same seeds and then over one failing case's schedules, and the fix is
# put back. A bug is detected by the contracts that fail with it and never on the tree as it stands
# (the baseline run).
#
# Fails (exit 1) when a bench run fails or prints no campaign, when a patch no longer reverts
# cleanly (refresh it from its fix), and under `--compare` when any (bug, contract, level,
# policy) present in both summaries hits fewer times than in the given one: the K-14 matched-seed
# comparison a change to `drive/` or `fakes/` records (`--rev` the commit before it, then the
# change, compared).
#
# `VOICE_SEEDS` (16384) campaign seeds, `VOICE_SETUPS` (1024) conversations on a failing setup and
# `VOICE_SCHEDULES` (512) schedules of a failing case, per policy. Results are keyed by the tree's
# identity (the commit, a hash of the diff and untracked files) and that configuration, under
# `VOICE_BENCH_OUT` (`$TMPDIR/lk-known-bugs/<identity>` by default): a run on another tree never
# reuses them, and only a complete run is kept. Writes `build.txt`, `summary.tsv` and `table.txt`
# there.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(git -C "$here" rev-parse --show-toplevel)"
crate="rust/crates/lk-agents-fuzz"

compare="" rev=""
while [ $# -gt 0 ]; do
    case "$1" in
        --compare)
            [ -s "${2:-}" ] || { echo "known_bugs: --compare ${2:-}: no such summary" >&2; exit 2; }
            compare="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
            shift 2
            ;;
        --rev)
            rev="$(git -C "$root" rev-parse --verify "${2:-}^{commit}")" \
                || { echo "known_bugs: --rev ${2:-}: no such commit" >&2; exit 2; }
            shift 2
            ;;
        *) break ;;
    esac
done
if [ $# -eq 0 ]; then
    set -- $(cd "$here/bugs" && ls ./*.patch | sed 's|^\./||; s|\.patch$||')
fi
for bug in "$@"; do
    [ -f "$here/bugs/$bug.patch" ] || { echo "known_bugs: no bugs/$bug.patch" >&2; exit 2; }
done

export VOICE_SEEDS="${VOICE_SEEDS:-16384}" VOICE_SCHEDULES="${VOICE_SCHEDULES:-512}"
export VOICE_SETUPS="${VOICE_SETUPS:-1024}"
config="seeds=$VOICE_SEEDS schedules=$VOICE_SCHEDULES setups=$VOICE_SETUPS \
budget=${VOICE_BUDGET:-1500} windows=${VOICE_WINDOWS:-8} first=${VOICE_FIRST_SEED:-0}"

# The tree's identity: the commit, and without `--rev` the diff and every untracked file under
# rust/.
files="$(mktemp)"
trap 'rm -f "$files"' EXIT
if [ -n "$rev" ]; then
    head="$rev"
    git -C "$root" ls-tree -r -z --name-only "$rev" -- rust > "$files"
    identity="$(echo "$rev $config" | shasum -a 256 | cut -c1-16)"
    dirty="none (--rev)"
else
    head="$(git -C "$root" rev-parse HEAD)"
    git -C "$root" ls-files -z --cached --others --exclude-standard -- rust > "$files"
    identity="$(
        {
            echo "$head $config"
            git -C "$root" diff HEAD --binary -- rust
            git -C "$root" ls-files -z --others --exclude-standard -- rust \
                | (cd "$root" && xargs -0 shasum -a 256 2>/dev/null)
        } | shasum -a 256 | cut -c1-16
    )"
    dirty="$(git -C "$root" status --porcelain -- rust | wc -l | tr -d ' ') paths under rust/"
fi
# One benchmark at a time: they share the copy and its target dir, apart from every checkout's
# (cargo names path crates' artifacts by their place in the workspace, so a copy building in the
# checkout's target dir and the checkout itself would overwrite each other's).
home="${VOICE_BENCH_HOME:-${TMPDIR:-/tmp}/lk-known-bugs}"
mkdir -p "$home"
mkdir "$home/lock" 2>/dev/null \
    || { echo "known_bugs: another benchmark holds $home/lock" >&2; exit 2; }
trap 'rm -f "$files"; rmdir "$home/lock"' EXIT
out="${VOICE_BENCH_OUT:-$home/$identity}"
mkdir -p "$out"
if [ -s "$out/identity" ] && [ "$(cat "$out/identity")" != "$identity" ]; then
    echo "known_bugs: $out holds results of another tree or configuration" >&2
    exit 2
fi
echo "$identity" > "$out/identity"
{
    echo "commit $head"
    echo "identity $identity"
    echo "config $config"
    echo "uncommitted $dirty"
    echo "rustc $(rustc --version)"
} > "$out/build.txt"
cat "$out/build.txt"

# The copy: what is tracked (or untracked) under rust/. Rewritten only where it differs, with a
# fresh mtime, so cargo rebuilds exactly what changed; what is no longer there is deleted.
tree="$home/tree"
mkdir -p "$tree"
if [ -n "$rev" ]; then
    staged="$home/rev"
    rm -rf "$staged"
    mkdir -p "$staged"
    git -C "$root" archive "$rev" rust | tar -x -C "$staged"
    source="$staged/"
else
    source="$root/"
fi
rsync -r --links --checksum --from0 --files-from="$files" "$source" "$tree/"
(cd "$tree" && find rust -path rust/target -prune -o \( -type f -o -type l \) -print | sort) \
    | comm -23 - <(tr '\0' '\n' < "$files" | sort) | (cd "$tree" && xargs rm -f)
export CARGO_TARGET_DIR="$home/target"

# `git apply` on the copy, which is no repository: paths are taken from the copy's root.
apply() { (cd "$tree" && GIT_DIR=/dev/null git apply "$@"); }

failed=()
bench() {
    local name=$1 binary
    (cd "$tree/rust" && cargo test --release -p lk-agents-fuzz --test pos --no-run \
        --message-format=json-diagnostic-rendered-ansi > "$out/$name.build" 2>&1) || true
    binary="$(python3 -c 'import json, sys
for line in open(sys.argv[1]):
    if not line.startswith("{"):
        continue
    m = json.loads(line)
    if m.get("reason") == "compiler-message" and m["message"]["level"] == "error":
        print(m["message"]["rendered"], file=sys.stderr)
    elif m.get("reason") == "compiler-artifact" and m["target"]["name"] == "pos" \
            and m["executable"]:
        print(m["executable"])' "$out/$name.build" 2>"$out/$name.errors")"
    if [ -z "$binary" ]; then
        echo "== $name: the build failed (see $out/$name.errors)"
        failed+=("$name")
        return
    fi
    # The binary, not `cargo test`: the target dir's lock is released while the bench runs.
    if ! (cd "$tree/$crate" && "$binary" --ignored --nocapture --exact bench) \
        > "$out/$name.part" 2>&1; then
        echo "== $name: the bench failed (see $out/$name.part)"
        failed+=("$name")
    elif ! grep -q $'^TSV\tcampaign\t' "$out/$name.part"; then
        echo "== $name: the bench ran no campaign (see $out/$name.part)"
        failed+=("$name")
    else
        mv "$out/$name.part" "$out/$name.txt"
    fi
}

if [ ! -s "$out/baseline.txt" ]; then
    echo "== baseline"
    bench baseline
fi
for bug in "$@"; do
    patch="$here/bugs/$bug.patch"
    if ! apply -R --check "$patch" 2>/dev/null; then
        echo "== $bug: STALE (the fix no longer reverts cleanly: refresh the patch)"
        failed+=("$bug")
        continue
    fi
    if [ -s "$out/$bug.txt" ]; then
        echo "== $bug: kept from an earlier run of this tree"
        continue
    fi
    echo "== $bug"
    apply -R "$patch"
    bench "$bug"
    apply "$patch"
done
if [ ! -s "$out/baseline.txt" ]; then
    echo "known_bugs: no baseline: nothing to tabulate" >&2
    exit 1
fi

# The table: per bug, the contracts it fails that the baseline never does, at three levels: the
# campaign (fresh seeds), the setup (its first failing case's setup, fresh conversations) and the
# program (that case's inputs, fresh schedules). Each cell: hits / tries, then the median over the
# windows of the tries to the first hit. The policies are the ones the bench ran.
present=()
for bug in "$@"; do
    [ -s "$out/$bug.txt" ] && present+=("$out/$bug.txt")
done
awk -F'\t' -v summary="$out/summary.tsv" '
    FNR == 1 {
        bug = FILENAME; sub(/.*\//, "", bug); sub(/\.txt$/, "", bug)
        if (bug != "baseline") bugs[++nb] = bug
    }
    $1 != "TSV" { next }
    bug == "baseline" && $2 != "campaign" && $2 != "finding" && $5 > 0 { base[$4] = 1 }
    bug == "baseline" && $2 == "campaign" {
        policies[++np] = $3; cost[$3] = $4; reached[$3] = $5; edges[$3] = $6; found[$3] = $7
    }
    bug != "baseline" && ($2 == "seeds" || $2 == "setup" || $2 == "program") {
        cell[bug, $4, $2, $3] = $5 "/" $6 " " $7
        hits[bug, $4, $2, $3] = $5
        tries[$2] = $6
        if ($2 == "program") contracts[bug, $4] = 1
    }
    END {
        split("seeds setup program", levels, " ")
        printf "%-26s %-34s %-8s", "bug", "contract", "level"
        for (p = 1; p <= np; p++) printf " %-16s", policies[p]
        printf "\n"
        printf "bug\tcontract\tlevel\tpolicy\thits\ttries\n" > summary
        for (b = 1; b <= nb; b++) {
            detected = 0
            for (key in contracts) {
                split(key, parts, SUBSEP)
                if (parts[1] != bugs[b] || (parts[2] in base)) continue
                detected = 1; rows++
                for (l = 1; l <= 3; l++) {
                    printf "%-26s %-34s %-8s", bugs[b], parts[2], levels[l]
                    for (p = 1; p <= np; p++) {
                        k = bugs[b] SUBSEP parts[2] SUBSEP levels[l] SUBSEP policies[p]
                        printf " %-16s", cell[k]
                        printf "%s\t%s\t%s\t%s\t%d\t%d\n", bugs[b], parts[2], levels[l],
                            policies[p], hits[k], tries[levels[l]] > summary
                        r = hits[k] / tries[levels[l]]
                        lg[l, p] += log(r > 0 ? r : 0.5 / tries[levels[l]])
                    }
                    printf "\n"
                }
            }
            if (!detected) printf "%-26s %s\n", bugs[b], "not detected by any policy"
        }
        if (rows) {
            printf "\ngeometric mean hit ratio over %d detected contracts (a zero counted as half a hit):\n", rows
            for (l = 1; l <= 3; l++) {
                printf "  %-8s", levels[l]
                for (p = 1; p <= np; p++) printf " %s %.5f", policies[p], exp(lg[l, p] / rows)
                printf "\n"
            }
        }
        printf "\nbaseline campaign: seeds/s, behaviours reached, edges, distinct findings\n"
        for (p = 1; p <= np; p++)
            printf "  %-10s %8s %6s %6s %4s\n", policies[p], cost[policies[p]],
                reached[policies[p]], edges[policies[p]], found[policies[p]]
    }
' "$out/baseline.txt" ${present[@]+"${present[@]}"} | tee "$out/table.txt"
echo "outputs: $out"

if [ -n "$compare" ]; then
    echo
    echo "matched-seed comparison against $compare:"
    if ! awk -F'\t' '
        FNR == 1 { next }
        NR == FNR { before[$1, $2, $3, $4] = $5; next }
        ($1, $2, $3, $4) in before {
            matched++
            if ($5 < before[$1, $2, $3, $4]) {
                printf "  FEWER HITS %s %s %s %s: %d -> %d of %d\n", $1, $2, $3, $4,
                    before[$1, $2, $3, $4], $5, $6
                fewer++
            }
        }
        END {
            printf "  %d matched cells, %d with fewer hits\n", matched, fewer
            exit (matched == 0 || fewer > 0)
        }
    ' "$compare" "$out/summary.tsv"; then
        failed+=("comparison")
    fi
fi

if [ ${#failed[@]} -gt 0 ]; then
    echo "known_bugs: FAILED: ${failed[*]}" >&2
    exit 1
fi
