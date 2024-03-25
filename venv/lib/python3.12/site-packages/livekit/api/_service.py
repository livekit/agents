from typing import Dict
import aiohttp
from abc import ABC
from .twirp_client import TwirpClient
from .access_token import AccessToken, VideoGrants

AUTHORIZATION = "authorization"


class Service(ABC):
    def __init__(
        self, session: aiohttp.ClientSession, host: str, api_key: str, api_secret: str
    ):
        self._client = TwirpClient(session, host, "livekit")
        self.api_key = api_key
        self.api_secret = api_secret

    def _auth_header(self, grants: VideoGrants) -> Dict[str, str]:
        token = AccessToken(self.api_key, self.api_secret).with_grants(grants).to_jwt()

        headers = {}
        headers[AUTHORIZATION] = "Bearer {}".format(token)
        return headers
