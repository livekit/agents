# Copyright 2026 LiveKit, Inc.
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

from __future__ import annotations

from typing import Literal

AlibabaRealtimeModels = Literal["qwen-audio-3.1-realtime-plus"]
DEFAULT_MODEL: AlibabaRealtimeModels = "qwen-audio-3.1-realtime-plus"

AlibabaRegion = Literal["cn", "intl"]
DEFAULT_REGION: AlibabaRegion = "cn"

AlibabaVoices = Literal[
    "longanqian",
    "longanlingxin",
    "longanlufeng",
    "longanlingxi",
    "longanxiaoxin",
    "longanfengyue",
    "longanyuanfei",
    "longanqian_v3.1",
    "longanhuan_v3.1",
    "longanlingxin_v3.1",
    "longanfengyue_v3.1",
    "xunanchuan_v3.1",
    "beth_v3.1",
    "betty_v3.1",
    "cally_v3.1",
]
DEFAULT_VOICE: AlibabaVoices = "longanqian"

REALTIME_BASE_URLS: dict[AlibabaRegion, str] = {
    "cn": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
    "intl": "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
}

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000


def get_realtime_url(
    *,
    region: AlibabaRegion = DEFAULT_REGION,
    base_url: str | None = None,
    workspace_id: str | None = None,
) -> str:
    """Resolve the DashScope endpoint; the parent adds the model query parameter."""
    if base_url:
        endpoint = base_url.rstrip("/")
    elif workspace_id:
        domain_region = "ap-southeast-1" if region == "intl" else "cn-beijing"
        endpoint = f"wss://{workspace_id}.{domain_region}.maas.aliyuncs.com/api-ws/v1/realtime"
    else:
        endpoint = REALTIME_BASE_URLS.get(region, REALTIME_BASE_URLS[DEFAULT_REGION])

    return endpoint
