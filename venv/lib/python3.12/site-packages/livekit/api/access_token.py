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

import calendar
import dataclasses
import re
import datetime
import os
import jwt
from typing import Optional, List

DEFAULT_TTL = datetime.timedelta(hours=6)
DEFAULT_LEEWAY = datetime.timedelta(minutes=1)


@dataclasses.dataclass
class VideoGrants:
    # actions on rooms
    room_create: bool = False
    room_list: bool = False
    room_record: bool = False

    # actions on a particular room
    room_admin: bool = False
    room_join: bool = False
    room: str = ""

    # permissions within a room
    can_publish: bool = True
    can_subscribe: bool = True
    can_publish_data: bool = True

    # TrackSource types that a participant may publish.
    # When set, it supercedes CanPublish. Only sources explicitly set here can be
    # published
    can_publish_sources: List[str] = dataclasses.field(default_factory=list)

    # by default, a participant is not allowed to update its own metadata
    can_update_own_metadata: bool = False

    # actions on ingresses
    ingress_admin: bool = False  # applies to all ingress

    # participant is not visible to other participants (useful when making bots)
    hidden: bool = False

    # indicates to the room that current participant is a recorder
    recorder: bool = False

    # indicates that the holder can register as an Agent framework worker
    # it is also set on all participants that are joining as Agent
    agent: bool = False


@dataclasses.dataclass
class Claims:
    identity: str = ""
    name: str = ""
    video: VideoGrants = dataclasses.field(default_factory=VideoGrants)
    metadata: str = ""
    sha256: str = ""


class AccessToken:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        api_key = api_key or os.getenv("LIVEKIT_API_KEY")
        api_secret = api_secret or os.getenv("LIVEKIT_API_SECRET")

        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret must be set")

        self.api_key = api_key  # iss
        self.api_secret = api_secret
        self.claims = Claims()

        # default jwt claims
        self.identity = ""  # sub
        self.ttl = DEFAULT_TTL  # exp

    def with_ttl(self, ttl: datetime.timedelta) -> "AccessToken":
        self.ttl = ttl
        return self

    def with_grants(self, grants: VideoGrants) -> "AccessToken":
        self.claims.video = grants
        return self

    def with_identity(self, identity: str) -> "AccessToken":
        self.identity = identity
        return self

    def with_name(self, name: str) -> "AccessToken":
        self.claims.name = name
        return self

    def with_metadata(self, metadata: str) -> "AccessToken":
        self.claims.metadata = metadata
        return self

    def with_sha256(self, sha256: str) -> "AccessToken":
        self.claims.sha256 = sha256
        return self

    def to_jwt(self) -> str:
        video = self.claims.video
        if video.room_join and (not self.identity or not video.room):
            raise ValueError("identity and room must be set when joining a room")

        claims = dataclasses.asdict(
            self.claims,
            dict_factory=lambda items: {snake_to_lower_camel(k): v for k, v in items},
        )
        claims.update(
            {
                "sub": self.identity,
                "iss": self.api_key,
                "nbf": calendar.timegm(datetime.datetime.utcnow().utctimetuple()),
                "exp": calendar.timegm(
                    (datetime.datetime.utcnow() + self.ttl).utctimetuple()
                ),
            }
        )

        return jwt.encode(claims, self.api_secret, algorithm="HS256")


class TokenVerifier:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        leeway: datetime.timedelta = DEFAULT_LEEWAY,
    ) -> None:
        api_key = api_key or os.getenv("LIVEKIT_API_KEY")
        api_secret = api_secret or os.getenv("LIVEKIT_API_SECRET")

        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret must be set")

        self.api_key = api_key
        self.api_secret = api_secret
        self._leeway = leeway

    def verify(self, token: str) -> Claims:
        claims = jwt.decode(
            token,
            self.api_secret,
            issuer=self.api_key,
            algorithms=["HS256"],
            leeway=self._leeway.total_seconds(),
        )

        video_dict = claims.get("video", dict())
        video_dict = {camel_to_snake(k): v for k, v in video_dict.items()}
        video_dict = {
            k: v for k, v in video_dict.items() if k in VideoGrants.__dataclass_fields__
        }
        video = VideoGrants(**video_dict)

        return Claims(
            identity=claims.get("sub", ""),
            name=claims.get("name", ""),
            video=video,
            metadata=claims.get("metadata", ""),
            sha256=claims.get("sha256", ""),
        )


def camel_to_snake(t: str):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", t).lower()


def snake_to_lower_camel(t: str):
    return "".join(
        word.capitalize() if i else word for i, word in enumerate(t.split("_"))
    )
