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
    *, client_id: Optional[str] = None, client_secret: Optional[str] = None
):
    if client_id is None or client_secret is None:
        try:
            client_id = os.environ["DIRECTAI_CLIENT_ID"]
            client_secret = os.environ["DIRECTAI_CLIENT_SECRET"]
        except KeyError:
            raise Exception(
                "DIRECTAI_CLIENT_ID or DIRECTAI_CLIENT_SECRET not set. Set them as environment variables or pass them in as arguments."
            )

    params = {"client_id": client_id, "client_secret": client_secret}
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL + "/token", data=params) as response:
            if response.status != 200:
                raise ValueError("Invalid DirectAI Credentials")
            return (await response.json())["access_token"]
