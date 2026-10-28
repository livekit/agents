"""Console view models; storage and chat records come from the SDK schema."""

from typing import Any

from pydantic import BaseModel

from livekit.agents._generated.storage import EventRow, InboxRow
from livekit.agents._generated.storage_run import LaneRow, TaskRow
from livekit.agents.llm import ChatItem


class StreamAvailability(BaseModel):
    url: str
    available: bool
    error: str | None = None


class Subjects(BaseModel):
    subs: list[str]


class ConsoleConfig(BaseModel):
    grant: Subjects
    streams: StreamAvailability


class Grant(BaseModel):
    token: str
    agent: str
    sub: str
    endpoint: str
    endpointName: str
    expiresAt: int


class SessionRow(BaseModel):
    sessionId: str
    agentName: str
    sub: str
    attributes: dict[str, str]
    region: str = ""
    databaseId: str
    createdAt: float
    lastActive: float
    expiresAt: float
    idleTtlSeconds: int


class Item(BaseModel):
    lane: str
    id: str
    seq: int
    type: str
    message_id: str | None
    payload: ChatItem


class DatabaseDump(BaseModel):
    databaseId: str
    tip: int
    tables: list[str]
    schemaVersion: int
    lanes: list[LaneRow]
    items: list[Item]
    events: list[EventRow]
    inbox: list[InboxRow]
    tasks: list[TaskRow]


class SessionDump(BaseModel):
    fetchedAt: int
    session: SessionRow
    database: DatabaseDump | None = None
    databaseError: str | None = None
    streams: StreamAvailability


class History(BaseModel):
    items: list[ChatItem]


class Sessions(BaseModel):
    sessions: list[SessionRow]
    before: str = ""


class CreatedSession(BaseModel):
    session: SessionRow


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    tip: int
    ms: int


class Example(BaseModel):
    name: str
    agent: str
    endpoint: str
    description: str
    voice: bool
    start: str
    dir: str
    url: str
    live: bool


class Examples(BaseModel):
    examples: list[Example]


class CallMetadata(BaseModel):
    session_id: str | None = None
    sub: str | None = None
    endpoint: str


class CallDispatch(BaseModel):
    agent_name: str
    metadata: str


class CallRoomConfig(BaseModel):
    agents: list[CallDispatch]


class CallRequest(BaseModel):
    room_name: str | None = None
    participant_name: str | None = None
    participant_identity: str | None = None
    participant_attributes: dict[str, str] | None = None
    room_config: CallRoomConfig


class CallToken(BaseModel):
    server_url: str
    participant_token: str
    session_id: str


class TitleRequest(BaseModel):
    sessionId: str
    firstMessage: str
