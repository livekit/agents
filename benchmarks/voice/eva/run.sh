#!/usr/bin/env bash
# Run EVA-Bench with a LiveKit Agents preset as the agent under test.
#
#   benchmarks/voice/eva/run.sh <preset> <run-name> [eva flags...]
#   benchmarks/voice/eva/run.sh baseline-gemma4 airline-smoke --debug
#   benchmarks/voice/eva/run.sh baseline-gemma4 airline --max-concurrent-conversations 8
#
# Starts the agent service (lk_voice_bench.ws_agent, in the repo's venv) and runs EVA's airline
# domain against it. Results go to benchmarks/voice/results/eva/<run-name>/, agent-side session
# logs to agent_logs/.
#
# Without Gemini (Vertex) and Bedrock access this deviates from EVA's published setup:
# - caller: EVA's OpenAI Realtime caller (gpt-realtime-1.5), not its ElevenLabs agents;
# - judges: OpenAI only. faithfulness is judged by gpt-5.2 (published: Claude Opus 4.6);
# - not run: agent_speech_fidelity (part of EVA-A pass) and tts_fidelity, which need the Gemini
#   audio judge, and the user_speech_fidelity validation check (EVA_SKIP_VALIDATION_METRICS).
#   EVA-A is therefore reported without its speech-fidelity condition.
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,17p' "$0"
  exit 2
fi
PRESET=$1
RUN=$2
shift 2

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$(dirname "$HERE")
ROOT=$(cd "$BENCH/../.." && pwd)
OUT="$BENCH/results/eva/$RUN"
PORT=${LK_BENCH_AGENT_PORT:-8765}
mkdir -p "$OUT"

# the agent under test, in the repo's own venv
(cd "$BENCH" && exec "$ROOT/.venv/bin/python" -m lk_voice_bench.ws_agent \
  --port "$PORT" --log-dir "$OUT/agent_logs") >"$OUT/ws_agent.log" 2>&1 &
AGENT_PID=$!
trap 'kill $AGENT_PID 2>/dev/null || true' EXIT
until grep -q listening "$OUT/ws_agent.log" 2>/dev/null; do
  kill -0 $AGENT_PID 2>/dev/null || { cat "$OUT/ws_agent.log"; exit 1; }
  sleep 0.5
done

export EVA_MODEL_LIST='[
  {"model_name": "gpt-5.2",
   "litellm_params": {"model": "openai/gpt-5.2", "api_key": "os.environ/OPENAI_API_KEY"},
   "model_info": {"base_model": "gpt-5.2"}},
  {"model_name": "gpt-5.2-medium",
   "litellm_params": {"model": "openai/gpt-5.2", "api_key": "os.environ/OPENAI_API_KEY",
                      "reasoning_effort": "medium"},
   "model_info": {"base_model": "gpt-5.2"}}
]'
export EVA_FRAMEWORK=livekit
export EVA_MODEL__S2S=livekit
# EVA requires an api_key for every s2s framework; the agent reads LiveKit credentials from its own
# environment, so this is a placeholder (and keeps a real secret out of EVA's saved config)
export EVA_MODEL__S2S_PARAMS="{\"model\": \"$PRESET\", \"url\": \"ws://127.0.0.1:$PORT/\", \"api_key\": \"unused-see-LIVEKIT_API_KEY\"}"
export EVA_USER_SIMULATOR__PROVIDER=openai_realtime
export EVA_METRICS=faithfulness,authentication_success,conversation_correctly_finished,response_speed,speakability,stt_wer,time_to_completion,tool_call_validity,transcription_accuracy_key_entities,turns_to_completion,task_completion,conciseness,conversation_progression,turn_taking
export EVA_METRIC_CONFIGS='{"faithfulness": {"judge_model": "gpt-5.2"}}'
export EVA_SKIP_VALIDATION_METRICS=user_speech_fidelity

cd "$BENCH/.vendor/eva"
"$BENCH/.venv-eva/bin/eva" \
  --domain airline \
  --run-id "$RUN" \
  --output-dir "$BENCH/results/eva" \
  "$@"
