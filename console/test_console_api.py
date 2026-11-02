"""Developer API boundary regressions; no running LiveKit services required."""

import importlib
import os
import sqlite3
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import jwt
import pytest
from starlette.testclient import TestClient

pytestmark = pytest.mark.unit

sys.path.insert(0, str(Path(__file__).resolve().parent))
with patch.dict(
    os.environ,
    {
        "LIVEKIT_URL": "ws://localhost:7880",
        "LIVEKIT_API_KEY": "devkey",
        "LIVEKIT_API_SECRET": "console-test-signing-secret-at-least-32-bytes",
        "CONSOLE_API_TOKEN": "console-test-secret",
        "CONSOLE_AGENTDB_ADMIN_URL": "http://admin.test",
    },
):
    console_api = importlib.import_module("console_api")


class ConsoleAccessMatrix(unittest.TestCase):
    def test_anonymous_requests_cannot_read_or_mint_credentials(self):
        with TestClient(console_api.app) as client:
            for path in [
                "/api/config",
                "/api/token?agent=desk&endpoint=chat",
                "/api/sessions/AT_x",
            ]:
                self.assertEqual(client.get(path).status_code, 401, path)

    def test_browser_grant_is_scoped_and_database_credentials_are_never_returned(self):
        with TestClient(
            console_api.app, headers={"Authorization": "Bearer console-test-secret"}
        ) as client:
            result = client.get("/api/token?sub=demo-user&agent=desk&endpoint=chat")
            self.assertEqual(result.status_code, 200)
            claims = jwt.decode(
                result.json()["token"],
                "console-test-signing-secret-at-least-32-bytes",
                algorithms=["HS256"],
            )
            self.assertEqual(claims["agentThread"]["scope"], "demo-user")
            self.assertTrue(claims["agentThread"]["read"])
            self.assertNotIn("agent", claims)
            self.assertNotIn("agentEndpoint", claims)
            self.assertEqual(client.get("/api/sessions/AT_x/db").status_code, 404)

    def test_first_call_creation_requires_agent_and_endpoint(self):
        with TestClient(
            console_api.app, headers={"Authorization": "Bearer console-test-secret"}
        ) as client:
            self.assertEqual(
                client.post("/api/sessions", json={"sub": "demo-user"}).status_code, 400
            )


class SnapshotAccess(unittest.IsolatedAsyncioTestCase):
    def test_snapshot_queries_cannot_write_or_access_other_databases(self):
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE sample(value TEXT)")
            db.execute("INSERT INTO sample VALUES (?)", ("original",))
            data = db.serialize()
        for sql in [
            "DELETE FROM sample RETURNING value",
            "UPDATE sample SET value = 'changed' RETURNING value",
            "SELECT value FROM sample; DELETE FROM sample",
            "ATTACH DATABASE ':memory:' AS other",
            "PRAGMA query_only=OFF",
            "PRAGMA writable_schema=ON",
            "SELECT load_extension('untrusted')",
        ]:
            with self.subTest(sql=sql), self.assertRaises(sqlite3.DatabaseError):
                console_api.query_snapshot(data, sql, 5, 7)
        result = console_api.query_snapshot(data, "SELECT value FROM sample", 5, 7)
        self.assertEqual(result["rows"], [["original"]])

    def test_snapshot_identifiers_cannot_change_queries(self):
        # Schema names are persisted input, including SQL-looking text and quote characters.
        for name in ['a"b', 'x" FROM sqlite_master --', 'x"; DROP TABLE victim; --']:
            for column_type, value, expected in [
                ("TEXT", "original value", "original value"),
                ("JSONB", '{"original": true}', {"original": True}),
                ("JSONB", None, None),
            ]:
                with self.subTest(name=name, column_type=column_type, value=value):
                    with sqlite3.connect(":memory:") as db:
                        db.row_factory = sqlite3.Row
                        # Build the fixture independently of the production quoting helper.
                        identifier = '"' + name.replace('"', '""') + '"'
                        db.execute("CREATE TABLE victim(value TEXT)")
                        db.execute(f"CREATE TABLE {identifier} ({identifier} {column_type})")
                        db.execute(f"INSERT INTO {identifier} VALUES (?)", (value,))
                        self.assertEqual(console_api.read_rows(db, name), [{name: expected}])
                        self.assertEqual(db.execute("SELECT COUNT(*) FROM victim").fetchone()[0], 0)

    async def test_existing_dump_rpc_retains_admin_credentials_and_downloads_without_them(self):
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE sample(value TEXT)")
            db.execute("INSERT INTO sample VALUES ('hello')")
            data = db.serialize()
        requests = []

        async def respond(request):
            requests.append(request)
            if request.url.host == "admin.test":
                self.assertEqual(request.url.path, "/twirp/livekit.AgentDBService/DumpDatabase")
                claims = jwt.decode(
                    request.headers["authorization"].split()[1],
                    "console-test-signing-secret-at-least-32-bytes",
                    algorithms=["HS256"],
                )
                self.assertTrue(claims["agent"]["databaseAdmin"])
                self.assertEqual(__import__("json").loads(request.content), {"database_id": "DB_1"})
                return httpx.Response(
                    200,
                    json={
                        "download_url": "https://download.test/export",
                        "tip": "7",
                        "expires_at": "9999999999",
                    },
                )
            self.assertNotIn("authorization", request.headers)
            return httpx.Response(200, content=data)

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with (
            patch.object(
                console_api.storage,
                "get_thread",
                AsyncMock(return_value=SimpleNamespace(database_id="DB_1")),
            ),
            patch.object(console_api.httpx, "AsyncClient", return_value=client),
        ):
            async with console_api.snapshot("AT_1") as (export, tip, _):
                result = console_api.query_snapshot(export, "SELECT value FROM sample", 5, tip)
                self.assertEqual(result["rows"], [["hello"]])
                self.assertEqual(result["tip"], 7)
                schema = console_api.query_snapshot(
                    export,
                    "SELECT m.name, p.name FROM sqlite_master m JOIN pragma_table_info(m.name) p WHERE m.type='table'",
                    5,
                    tip,
                )
                self.assertEqual(schema["rows"], [["sample", "value"]])
                for sql in [
                    "DELETE FROM sample",
                    "ATTACH DATABASE ':memory:' AS other",
                    "PRAGMA writable_schema=ON",
                ]:
                    with self.assertRaises(sqlite3.DatabaseError):
                        console_api.query_snapshot(export, sql, 5, tip)
        self.assertEqual(len(requests), 2)

    def test_malformed_storage_snapshot_fails_at_the_generated_model_boundary(self):
        from models import DatabaseDump
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            DatabaseDump.model_validate(
                {
                    "databaseId": "DB_1",
                    "tip": 1,
                    "tables": [],
                    "schemaVersion": 1,
                    "lanes": [{"lane": 5}],
                    "items": [],
                    "events": [],
                    "inbox": [],
                    "tasks": [],
                }
            )


if __name__ == "__main__":
    unittest.main()
