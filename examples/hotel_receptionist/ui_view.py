from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable

import apsw

from livekit import rtc

logger = logging.getLogger("hotel-receptionist.ui_view")

_RPC_PATCH = "sqlite_diff"
_RPC_SUBSCRIBE = "sqlite_diff:subscribe"
_RPC_REBASE = "sqlite_diff:rebase"
_BASE_STREAM_TOPIC = "sqlite_diff:base"


class UiView:
    def __init__(self, room: rtc.Room, connection: apsw.Connection) -> None:
        self._room = room
        self._conn = connection
        self._session: apsw.Session | None = None
        self._version = 0
        self._schema_sql = ""
        self._subscribers: set[str] = set()
        self._lock = asyncio.Lock()
        self._registered_rpc_methods: set[str] = set()
        self._disconnect_handler: Callable[[rtc.RemoteParticipant], None] | None = None

    async def start(self) -> None:
        self._session = self._open_session()
        self._schema_sql = "\n".join(
            r[0]
            for r in self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY rootpage"
            )
        )
        lp = self._room.local_participant
        lp.register_rpc_method(_RPC_SUBSCRIBE, self._handle_subscribe)
        self._registered_rpc_methods.add(_RPC_SUBSCRIBE)
        lp.register_rpc_method(_RPC_REBASE, self._handle_rebase)
        self._registered_rpc_methods.add(_RPC_REBASE)

        def _on_disconnect(p: rtc.RemoteParticipant) -> None:
            self._subscribers.discard(p.identity)

        self._room.on("participant_disconnected", _on_disconnect)
        self._disconnect_handler = _on_disconnect

        logger.info("ui_view started")

    async def on_change(self) -> None:
        async with self._lock:
            if self._session is None:
                return
            changeset = self._session.changeset()
            if not changeset:
                return
            self._session.close()
            self._session = self._open_session()
            self._version += 1
            tables = sorted({change.name for change in apsw.Changeset.iter(changeset)})
            await self._send_patch(self._version, tables, changeset)

    async def aclose(self) -> None:
        async with self._lock:
            if self._session is not None:
                self._session.close()
                self._session = None
            registered_rpc_methods = self._registered_rpc_methods
            self._registered_rpc_methods = set()
            disconnect_handler = self._disconnect_handler
            self._disconnect_handler = None
            self._subscribers.clear()

        if disconnect_handler is not None:
            self._room.off("participant_disconnected", disconnect_handler)

        lp = self._room.local_participant
        for method in registered_rpc_methods:
            try:
                lp.unregister_rpc_method(method)
            except Exception:
                pass

    def _open_session(self) -> apsw.Session:
        s = apsw.Session(self._conn, "main")
        s.attach(None)
        return s

    async def _handle_subscribe(self, data: rtc.RpcInvocationData) -> str:
        await self._stream_base_to(data.caller_identity)
        return json.dumps({"version": self._version})

    async def _handle_rebase(self, data: rtc.RpcInvocationData) -> str:
        await self._stream_base_to(data.caller_identity)
        return json.dumps({"version": self._version})

    async def _stream_base_to(self, identity: str) -> None:
        async with self._lock:
            self._version += 1
            version = self._version
            if self._session is not None:
                self._session.close()
            db_bytes: bytes = await asyncio.to_thread(self._conn.serialize, "main")
            objects = [
                (t, n)
                for t, n in self._conn.execute(
                    "SELECT type, name FROM sqlite_master "
                    "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
                    "ORDER BY type, name"
                )
            ]
            logger.info("base v%d objects: %s", self._version, objects)
            self._session = self._open_session()
            self._subscribers.add(identity)
        try:
            logger.info("streaming base v%d (%d bytes) to %s", version, len(db_bytes), identity)
            writer = await self._room.local_participant.stream_bytes(
                name=f"sqlite_diff_base_v{version}.sqlite",
                mime_type="application/vnd.sqlite3",
                topic=_BASE_STREAM_TOPIC,
                attributes={"schema_sql": self._schema_sql, "version": str(version)},
                total_size=len(db_bytes),
                destination_identities=[identity],
            )
            await writer.write(db_bytes)
            await writer.aclose()
        except Exception:
            logger.exception("failed to stream base to %s", identity)
            self._subscribers.discard(identity)

    async def _send_patch(self, version: int, tables_changed: list[str], changeset: bytes) -> None:
        if not self._subscribers:
            return
        payload = json.dumps(
            {
                "type": "patch",
                "version": version,
                "tables_changed": tables_changed,
                "changeset_b64": base64.b64encode(changeset).decode("ascii"),
            },
            separators=(",", ":"),
        )
        for identity in list(self._subscribers):
            try:
                await self._room.local_participant.perform_rpc(
                    destination_identity=identity, method=_RPC_PATCH, payload=payload
                )
            except Exception as e:
                logger.warning("patch v%d to %s failed: %s", version, identity, e)
