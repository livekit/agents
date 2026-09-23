#!/usr/bin/env bash
# Regenerates livekit_agentdb_pb2 from livekit/protocol. These files move to livekit-protocol
# once a release ships the agentdb module; until then they are vendored here.
#
# protoc 29 emits protobuf 5.29 gencode, the same floor livekit-protocol's modules carry, so
# the vendored module runs on protobuf 5.29+ and 6.x alike.
set -euo pipefail

PROTOCOL_DIR="${PROTOCOL_DIR:-$HOME/code/protocol}"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"

uvx --from 'grpcio-tools==1.71.0' python -m grpc_tools.protoc \
  -I "$PROTOCOL_DIR/protobufs" \
  --python_out="$OUT_DIR" \
  --pyi_out="$OUT_DIR" \
  "$PROTOCOL_DIR/protobufs/livekit_agentdb.proto"
