#!/usr/bin/env bash
# The storage and runtime contracts against a real AgentDB; the SQLite ones run in the launch gate.
set -euo pipefail
for name in LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET; do
    [[ -n ${!name:-} ]] || { echo "AgentDB backend contracts require $name" >&2; exit 2; }
done
cd "$(dirname "$0")"
exec cargo nextest run --locked -p lk-agents-storage -p lk-agents-runtime --lib \
    --test storage --test runtime --no-tests fail --run-ignored only -E 'test(agentdb::)'
