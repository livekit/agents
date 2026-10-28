"""Console view models; storage and chat records come from the SDK schema."""

from typing import Any

from pydantic import BaseModel

from livekit.agents._generated.storage import EventRow, InboxRow
from livekit.agents._generated.storage_run import LaneRow, TaskRow
from livekit.agents.llm import ChatItem


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


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    tip: int
    ms: int
