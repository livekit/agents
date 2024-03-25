import aiohttp
from livekit.protocol import ingress as proto_ingress
from ._service import Service
from .access_token import VideoGrants

SVC = "Ingress"


class IngressService(Service):
    def __init__(
        self, session: aiohttp.ClientSession, url: str, api_key: str, api_secret: str
    ):
        super().__init__(session, url, api_key, api_secret)

    async def create_ingress(
        self, create: proto_ingress.CreateIngressRequest
    ) -> proto_ingress.IngressInfo:
        return await self._client.request(
            SVC,
            "CreateIngress",
            create,
            self._auth_header(VideoGrants(ingress_admin=True)),
            proto_ingress.IngressInfo,
        )

    async def update_ingress(
        self, update: proto_ingress.UpdateIngressRequest
    ) -> proto_ingress.IngressInfo:
        return await self._client.request(
            SVC,
            "UpdateIngress",
            update,
            self._auth_header(VideoGrants(ingress_admin=True)),
            proto_ingress.IngressInfo,
        )

    async def list_ingress(
        self, list: proto_ingress.ListIngressRequest
    ) -> proto_ingress.ListIngressResponse:
        return await self._client.request(
            SVC,
            "ListIngress",
            list,
            self._auth_header(VideoGrants(ingress_admin=True)),
            proto_ingress.ListIngressResponse,
        )

    async def delete_ingress(
        self, delete: proto_ingress.DeleteIngressRequest
    ) -> proto_ingress.IngressInfo:
        return await self._client.request(
            SVC,
            "DeleteIngress",
            delete,
            self._auth_header(VideoGrants(ingress_admin=True)),
            proto_ingress.IngressInfo,
        )
