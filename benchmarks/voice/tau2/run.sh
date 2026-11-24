#!/usr/bin/env bash
# Run tau2 voice with a LiveKit Agents preset as the agent under test.
#
#   benchmarks/voice/tau2/run.sh <preset> <run-name> [tau2 run flags...]
#   benchmarks/voice/tau2/run.sh baseline-gemma4 smoke --domain retail --num-tasks 2
#
# Runs the released voice-v1 frame (English; airline 50, retail 114, telecom 114 tasks; --domain
# narrows it). voice-v1 refuses task samples, so a run with --num-tasks or --task-ids becomes a
# custom run (--benchmark none) with voice-v1's caller, judge and limits pinned to match.
# tau2 re-runs a call whose simulated caller hallucinated, judged by --review-model (Sierra's
# default is claude-opus-4-5; this defaults to gpt-5.5 on OPENAI_API_KEY, override with
# TAU2_REVIEW_MODEL).
# Results go to benchmarks/voice/results/tau2/<run-name>/, agent-side session logs to agent_logs/.
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,8p' "$0"
  exit 2
fi
PRESET=$1
RUN=$2
shift 2

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$(dirname "$HERE")
OUT="$BENCH/results/tau2/$RUN"
mkdir -p "$OUT"

# caller voice IDs (from setup_voices), kept out of git; a variable already set wins, so one
# run can swap a persona's voice: TAU2_VOICE_ID_EN_MAMADOU_DIALLO=<id> tau2/run.sh ...
if [ -f "$BENCH/.env.tau2.local" ]; then
  while IFS='=' read -r key value; do
    case "$key" in '' | \#*) continue ;; esac
    [ -z "${!key:-}" ] && export "$key=$value"
  done <"$BENCH/.env.tau2.local"
fi
export ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY:-${ELEVEN_API_KEY:-}}
export LK_BENCH_LOG_DIR="$OUT/agent_logs"

FRAME=(--benchmark voice-v1)
for arg in "$@"; do
  case "$arg" in
    --num-tasks | --task-ids)
      FRAME=(
        --benchmark none
        --user-llm openai/gpt-5.5
        --user-llm-args '{"temperature": 0.0, "reasoning_effort": "low"}'
        --communicate-judge '{"model": "gpt-5.5", "model_args": {"reasoning_effort": "low"}}'
        --speech-complexity regular
        --max-steps-seconds 2400
      )
      ;;
  esac
done

cd "$BENCH/.vendor/tau2-bench"
exec "$BENCH/.venv-tau2/bin/tau2" run \
  --audio-native \
  "${FRAME[@]}" \
  --audio-native-provider livekit_agents \
  --audio-native-model "$PRESET" \
  --save-to "$OUT" \
  --review-model "${TAU2_REVIEW_MODEL:-gpt-5.5}" \
  "$@"
