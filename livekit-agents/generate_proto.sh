#!/bin/bash
# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This script requires protobuf-compiler and https://github.com/nipunn1313/mypy-protobuf

set -e

API_PROTOCOL=./protocol
API_OUT_PYTHON=./livekit/agents/_proto

mkdir -p $API_OUT_PYTHON
 
# api

protoc \
    -I=$API_PROTOCOL \
    --python_out=$API_OUT_PYTHON \
    --mypy_out=$API_OUT_PYTHON \
    $API_PROTOCOL/livekit_agent.proto \
    $API_PROTOCOL/livekit_models.proto


touch -a "$API_OUT_PYTHON/__init__.py"

for f in "$API_OUT_PYTHON"/*.py "$API_OUT_PYTHON"/*.pyi; do
    perl -i -pe 's|^(import (livekit_egress_pb2\|livekit_room_pb2\|livekit_webhook_pb2\|livekit_ingress_pb2\|livekit_models_pb2\|livekit_agent_pb2))|from . $1|g' "$f"
done
