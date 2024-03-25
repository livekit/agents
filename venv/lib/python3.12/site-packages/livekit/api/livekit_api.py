import aiohttp
import os
from .room_service import RoomService
from .egress_service import EgressService
from .ingress_service import IngressService
from typing import Optional


class LiveKitAPI:
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        timeout: aiohttp.ClientTimeout = aiohttp.ClientTimeout(total=60),  # 60 seconds
    ):
        url = url or os.getenv("LIVEKIT_URL")
        api_key = api_key or os.getenv("LIVEKIT_API_KEY")
        api_secret = api_secret or os.getenv("LIVEKIT_API_SECRET")

        if not url:
            raise ValueError("url must be set")

        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret must be set")

        self._session = aiohttp.ClientSession(timeout=timeout)
        self._room = RoomService(self._session, url, api_key, api_secret)
        self._ingress = IngressService(self._session, url, api_key, api_secret)
        self._egress = EgressService(self._session, url, api_key, api_secret)

    @property
    def room(self):
        return self._room

    @property
    def ingress(self):
        return self._ingress

    @property
    def egress(self):
        return self._egress

    async def aclose(self):
        await self._session.close()
