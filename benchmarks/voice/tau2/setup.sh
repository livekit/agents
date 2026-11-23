#!/usr/bin/env bash
# Set up tau2-bench (pinned) with the livekit_agents provider, in its own venv.
#
#   benchmarks/voice/tau2/setup.sh
#
# Creates benchmarks/voice/.vendor/tau2-bench and benchmarks/voice/.venv-tau2. livekit-agents and
# lk_voice_bench are installed editable from this checkout, so SDK changes apply without a reinstall.
set -euo pipefail

TAU2_REPO=https://github.com/sierra-research/tau2-bench
TAU2_COMMIT=768f3d8fe8528b4f32a5096f9ed008ae50f48ead  # 2026-11-20, tau2 2.0.0

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$(dirname "$HERE")
ROOT=$(cd "$BENCH/../.." && pwd)
VENDOR="$BENCH/.vendor/tau2-bench"
VENV="$BENCH/.venv-tau2"
PATCH="$HERE/register-livekit-agents.patch"

if [ ! -d "$VENDOR/.git" ]; then
  git clone --quiet "$TAU2_REPO" "$VENDOR"
fi
git -C "$VENDOR" fetch --quiet origin
git -C "$VENDOR" checkout --quiet --force "$TAU2_COMMIT"
git -C "$VENDOR" reset --quiet --hard "$TAU2_COMMIT"
git -C "$VENDOR" apply "$PATCH"
echo "tau2-bench $(git -C "$VENDOR" rev-parse --short HEAD) + livekit_agents provider"

[ -d "$VENV" ] || uv venv --quiet --python 3.13 "$VENV"
# tau2's voice extra needs pyaudio, which builds against PortAudio
if [ ! -f /usr/include/portaudio.h ] && command -v apt-get >/dev/null; then
  echo "installing PortAudio headers (portaudio19-dev)"
  apt-get install -y -q portaudio19-dev >/dev/null 2>&1 ||
    { apt-get update -q >/dev/null 2>&1 && apt-get install -y -q portaudio19-dev >/dev/null; }
fi
# tau2 still imports audioop, which Python 3.13 dropped
VIRTUAL_ENV="$VENV" uv pip install --quiet \
  -e "$VENDOR[voice]" \
  "audioop-lts; python_version >= '3.13'" \
  -e "$ROOT/livekit-agents" \
  -e "$BENCH"
echo "venv: $VENV  (run tau2 with $VENV/bin/tau2)"
