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

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta

DEFAULT_API_URL = "https://developers.synthesia.io"
DEFAULT_JOIN_TIMEOUT = 30.0
DEFAULT_SWAP_TIMEOUT = 15.0
AVATAR_IDENTITY = "synthesia-avatar-agent"
AVATAR_NAME = "Synthesia avatar"
MAX_AVATAR_IDS = 5

# Lifetime of the room-join token minted for the avatar worker. Long enough to
# cover the full session and any reconnects.
TOKEN_TTL = timedelta(hours=6)


@dataclass
class AvatarConfig:
    """The avatars to render in the room.

    ``avatar_ids`` holds one to five gallery ids of Synthesia avatars available
    to your workspace. The first id is the active avatar; the rest are
    available for swapping in during the session. Normalized to a tuple.
    """

    avatar_ids: Sequence[str]

    def __post_init__(self) -> None:
        if isinstance(self.avatar_ids, (str, bytes)):
            raise ValueError("avatar_ids must be a list of ids, not a single string")
        self.avatar_ids = tuple(self.avatar_ids)
        if not 1 <= len(self.avatar_ids) <= MAX_AVATAR_IDS:
            raise ValueError(
                f"avatar_ids must contain between 1 and {MAX_AVATAR_IDS} ids, "
                f"got {len(self.avatar_ids)}"
            )


@dataclass(frozen=True)
class SessionConfig:
    avatar_ids: tuple[str, ...]
    api_key: str = field(repr=False)
    api_url: str
    join_timeout: float


@dataclass(frozen=True)
class StartSessionRequest:
    avatar_ids: list[str]
    livekit_url: str
    lk_token: str = field(repr=False)


@dataclass(frozen=True)
class StartSessionResponse:
    session_id: str
