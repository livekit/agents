#!/usr/bin/env bash
# Set up EVA-Bench (pinned) with the livekit framework, in its own venv.
#
#   benchmarks/voice/eva/setup.sh
#
# Creates benchmarks/voice/.vendor/eva and benchmarks/voice/.venv-eva. EVA cannot share a venv
# with livekit-agents (its litellm pin needs openai<3), so the agent runs separately
# (lk_voice_bench.ws_agent, in the repo's own venv) and EVA gets only the bridge server,
# symlinked in so edits to benchmarks/voice/eva/livekit_server.py apply directly.
set -euo pipefail

EVA_REPO=https://github.com/ServiceNow/eva
EVA_COMMIT=5a79590f3fc68f5c27a844729ca5160c54e3526b  # 2026-11-18

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$(dirname "$HERE")
VENDOR="$BENCH/.vendor/eva"
VENV="$BENCH/.venv-eva"
PATCH="$HERE/register-livekit.patch"

if [ ! -d "$VENDOR/.git" ]; then
  git clone --quiet "$EVA_REPO" "$VENDOR"
fi
git -C "$VENDOR" fetch --quiet origin
git -C "$VENDOR" checkout --quiet --force "$EVA_COMMIT"
git -C "$VENDOR" reset --quiet --hard "$EVA_COMMIT"
git -C "$VENDOR" apply "$PATCH"
ln -sf "$HERE/livekit_server.py" "$VENDOR/src/eva/assistant/livekit_server.py"
echo "eva $(git -C "$VENDOR" rev-parse --short HEAD) + livekit framework"

# EVA's own lockfile, so the venv is the one EVA is tested with
(cd "$VENDOR" && UV_PROJECT_ENVIRONMENT="$VENV" uv sync --quiet --frozen)
echo "venv: $VENV  (run EVA with $VENV/bin/eva)"
