#!/usr/bin/env bash
# vendored until a livekit-protocol release ships the agentdb module, then imported from there
# protoc 29 emits protobuf 5.29 gencode, which runs on protobuf 5.29+ and 6.x alike
set -euo pipefail

PROTOCOL_DIR="${PROTOCOL_DIR:-$HOME/code/protocol}"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"

uvx --from 'grpcio-tools==1.71.0' python -m grpc_tools.protoc \
  -I "$PROTOCOL_DIR/protobufs" \
  --python_out="$OUT_DIR" \
  --pyi_out="$OUT_DIR" \
  "$PROTOCOL_DIR/protobufs/livekit_agentdb.proto"
