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

import os
import aiohttp
from typing import Optional

API_URL = "https://api.alpha.directai.io"


async def generate_token(
    *,
    http_session: aiohttp.ClientSession,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
):
    params = {"client_id": client_id, "client_secret": client_secret}

    async with http_session.post("/token", json=params) as response:
        if response.status != 200:
            raise ValueError("Invalid DirectAI Credentials")
        return (await response.json())["access_token"]
