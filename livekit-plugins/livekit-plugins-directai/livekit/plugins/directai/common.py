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
