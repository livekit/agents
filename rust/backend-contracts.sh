#!/usr/bin/env bash
# The applicable shared runtime contracts; AgentDB is explicitly credentialed, never silently skipped.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && $2 != --check ) ]]; then
    echo "usage: $0 sqlite|agentdb [--check]" >&2
    exit 2
fi
backend=$1
case "$backend" in
    sqlite) ;;
    agentdb)
        for name in LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET; do
            if [[ -z ${!name:-} ]]; then
                echo "AgentDB backend contracts require $name" >&2
                exit 2
            fi
        done
        ;;
    *) echo "unknown backend: $backend" >&2; exit 2 ;;
esac
[[ ${2:-} != --check ]] || exit 0

cd "$(dirname "$0")"
args=(
    --locked -p lk-agents-storage -p lk-agents-runtime --lib
    --test lifecycle --test runs --test lease --test process --test schema --test memory
    --test storage_context --test storage_voice --test a2a_turns --test a2a_door
    --test a2a_attachment_boundary
    --no-tests fail
)
if [[ $backend == agentdb ]]; then
    args+=(--run-ignored only -E 'test(agentdb::)')
else
    args+=(-E 'kind(test) | package(lk-agents-storage) | test(storage::) | test(a2a_) | test(agent_run::) | test(sse::)')
fi
exec cargo nextest run "${args[@]}"
