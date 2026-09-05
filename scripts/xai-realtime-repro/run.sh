#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  unit        Run hermetic xAI realtime unit tests (delete-ack coverage)
  ref-probe   Live mid-session nested \$ref tools + response.create probe
  recycle     Websocket recycle demo (max_session_duration); pass --dry-log offline
  context     Multi-turn context retention probe
  all         unit, then live probes (live steps need XAI_API_KEY)

Env:
  XAI_API_KEY            required for ref-probe, recycle (live), context
  XAI_REALTIME_MODEL     default grok-voice-latest
  XAI_RECYCLE_SECONDS    default 45
  XAI_RECYCLE_DRY=1      same as recycle --dry-log
EOF
}

run_unit() {
  cd "$ROOT"
  uv run pytest tests/test_realtime/test_xai_realtime_model.py -q
}

cmd="${1:-}"
shift || true

case "$cmd" in
  unit)
    run_unit
    ;;
  ref-probe)
    cd "$ROOT"
    uv run python "$DIR/ref_probe.py"
    ;;
  recycle)
    cd "$ROOT"
    uv run python "$DIR/recycle_demo.py" "$@"
    ;;
  context)
    cd "$ROOT"
    uv run python "$DIR/context_probe.py"
    ;;
  all)
    run_unit
    cd "$ROOT"
    uv run python "$DIR/ref_probe.py"
    uv run python "$DIR/recycle_demo.py" "$@"
    uv run python "$DIR/context_probe.py"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
