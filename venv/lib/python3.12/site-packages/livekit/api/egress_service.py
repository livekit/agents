import aiohttp
from livekit.protocol import egress as proto_egress
from ._service import Service
from .access_token import VideoGrants

SVC = "Egress"


class EgressService(Service):
    def __init__(
        self, session: aiohttp.ClientSession, url: str, api_key: str, api_secret: str
    ):
        super().__init__(session, url, api_key, api_secret)

    async def start_room_composite_egress(
        self, start: proto_egress.RoomCompositeEgressRequest
    ) -> proto_egress.EgressInfo:
        return await self._client.request(
            SVC,
            "StartRoomCompositeEgress",
            start,
            self._auth_header(VideoGrants(room_record=True)),
            proto_egress.EgressInfo,
        )

    async def start_web_egress(
        self, start: proto_egress.WebEgressRequest
    ) -> proto_egress.EgressInfo:
        return await self._client.request(
            SVC,
            "StartWebEgress",
            start,
            self._auth_header(VideoGrants(room_record=True)),
            proto_egress.EgressInfo,
        )

    async def start_participant_egress(
        self, start: proto_egress.ParticipantEgressRequest
    ) -> proto_egress.EgressInfo:
        return await self._client.request(
            SVC,
            "StartParticipantEgress",
            start,
            self._auth_header(VideoGrants(room_record=True)),
            proto_egress.EgressInfo,
        )

    async def start_track_composite_egress(
        self, start: proto_egress.TrackCompositeEgressRequest
    ) -> proto_egress.EgressInfo:
        return await self._client.request(
            SVC,
            "StartTrackCompositeEgress",
            start,
            self._auth_header(VideoGrants(room_record=True)),
            proto_egress.EgressInfo,
        )

    async def start_track_egress(
        self, start: proto_egress.TrackEgressRequest
    ) -> proto_egress.EgressInfo:
        return await self._client.request(
            SVC,
            "StartTrackEgress",
            start,
            self._auth_header(VideoGrants(room_record=True)),
            proto_egress.EgressInfo,
        )

    async def update_layout(
        self, update: proto_egress.UpdateLayoutRequest
    ) -> proto_egress.EgressInfo:
        return await self._client.request(
            SVC,
            "UpdateLayout",
            update,
            self._auth_header(VideoGrants(room_record=True)),
            proto_egress.EgressInfo,
        )

    async def update_stream(
        self, update: proto_egress.UpdateStreamRequest
    ) -> proto_egress.EgressInfo:
        return await self._client.request(
            SVC,
            "UpdateStream",
            update,
            self._auth_header(VideoGrants(room_record=True)),
            proto_egress.EgressInfo,
        )

    async def list_egress(
        self, list: proto_egress.ListEgressRequest
    ) -> proto_egress.ListEgressResponse:
        return await self._client.request(
            SVC,
            "ListEgress",
            list,
            self._auth_header(VideoGrants(room_record=True)),
            proto_egress.ListEgressResponse,
        )

    async def stop_egress(
        self, stop: proto_egress.StopEgressRequest
    ) -> proto_egress.EgressInfo:
        return await self._client.request(
            SVC,
            "StopEgress",
            stop,
            self._auth_header(VideoGrants(room_record=True)),
            proto_egress.EgressInfo,
        )
