"""
Mint LiveKit participant tokens, dispatch the AiTwin demo agent, and issue AiTwin auth tokens.

Local-dev only: binds to TOKEN_SERVER_HOST (default 127.0.0.1). When TOKEN_SERVER_SECRET
is set, callers must send matching X-Token-Server-Secret. Room names are always
server-generated so callers cannot join arbitrary existing sessions.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from uuid import uuid4

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from livekit import api

load_dotenv()

app = Flask(__name__)
CORS(app)


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name} (set it in examples/aitwin/.env)")
    return value


LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
AGENT_NAME = _require_env("AGENT_NAME")

AITWIN_CLIENT_ID = os.getenv("AITWIN_CLIENT_ID")
AITWIN_CLIENT_SECRET = os.getenv("AITWIN_CLIENT_SECRET")
AITWIN_AUTH_TOKEN_URL = _require_env("AITWIN_AUTH_TOKEN_URL")
AITWIN_TWIN_ID = _require_env("AITWIN_TWIN_ID")
AITWIN_API_BASE = _require_env("AITWIN_API_BASE")
TOKEN_SERVER_PORT = int(_require_env("TOKEN_SERVER_PORT"))
TOKEN_SERVER_HOST = (os.getenv("TOKEN_SERVER_HOST") or "127.0.0.1").strip()
TOKEN_SERVER_SECRET = (os.getenv("TOKEN_SERVER_SECRET") or "").strip()


async def create_room_and_dispatch_agent(room_name: str) -> None:
    lkapi = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    try:
        room = await lkapi.room.create_room(api.CreateRoomRequest(name=room_name))
        print(f"Room ready: {room.name} ({room.sid})")
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(room=room_name, agent_name=AGENT_NAME)
        )
        print(f"Agent dispatch: {dispatch}")
    finally:
        await lkapi.aclose()


def _mint_aitwin_auth_token(*, user_id: str, user_name: str) -> str:
    if not AITWIN_CLIENT_ID or not AITWIN_CLIENT_SECRET:
        raise RuntimeError("AITWIN_CLIENT_ID / AITWIN_CLIENT_SECRET missing")

    response = requests.post(
        AITWIN_AUTH_TOKEN_URL,
        headers={
            "Content-Type": "application/json",
            "Client-Id": AITWIN_CLIENT_ID,
            "Client-Secret": AITWIN_CLIENT_SECRET,
        },
        json={
            "userId": user_id,
            "userName": user_name,
            "expiresIn": 60 * 60,
        },
        timeout=20,
    )
    data = response.json() if response.content else {}
    if not response.ok or not data.get("success") or not data.get("authToken"):
        raise RuntimeError(data.get("error") or f"AiTwin auth failed ({response.status_code})")
    return data["authToken"]


def _authorize_token_request() -> tuple[dict | None, int | None]:
    """Return (error_json, status) if unauthorized; else (None, None)."""
    if not TOKEN_SERVER_SECRET:
        return None, None
    header = (request.headers.get("X-Token-Server-Secret") or "").strip()
    body = request.get_json(silent=True) or {}
    body_secret = body.get("secret") if isinstance(body.get("secret"), str) else ""
    provided = header or body_secret.strip()
    if provided != TOKEN_SERVER_SECRET:
        return {"error": "unauthorized"}, 401
    return None, None


@app.post("/token")
def generate_token():
    auth_err, auth_status = _authorize_token_request()
    if auth_err is not None:
        return jsonify(auth_err), auth_status

    if not LIVEKIT_URL or not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        return jsonify({"error": "LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET missing"}), 500

    data = request.get_json(silent=True) or {}
    # Always mint a fresh room — never honor client-supplied room names.
    room_name = f"aitwin-demo-{uuid4().hex[:8]}"
    requested = data.get("identity")
    identity = (
        requested.strip()
        if isinstance(requested, str) and requested.strip()
        else f"browser-{uuid4().hex[:8]}"
    )

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(timedelta(hours=1))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )

    try:
        asyncio.run(create_room_and_dispatch_agent(room_name))
    except Exception as exc:
        print(f"Warning: agent dispatch failed: {exc}")

    aitwin_auth_token = None
    aitwin_error = None
    try:
        aitwin_auth_token = _mint_aitwin_auth_token(
            user_id=identity,
            user_name=identity,
        )
    except Exception as exc:
        aitwin_error = str(exc)
        print(f"Warning: AiTwin auth token failed: {exc}")

    return jsonify(
        {
            "token": token,
            "room": room_name,
            "roomName": room_name,
            "identity": identity,
            "url": LIVEKIT_URL,
            "aitwinAuthToken": aitwin_auth_token,
            "aitwinTwinId": AITWIN_TWIN_ID,
            "aitwinApiBase": AITWIN_API_BASE,
            "aitwinError": aitwin_error,
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "agent_name": AGENT_NAME,
            "livekit_configured": bool(LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET),
            "aitwin_configured": bool(AITWIN_CLIENT_ID and AITWIN_CLIENT_SECRET),
            "aitwin_twin_id": AITWIN_TWIN_ID,
            "token_secret_required": bool(TOKEN_SERVER_SECRET),
            "bind_host": TOKEN_SERVER_HOST,
        }
    )


if __name__ == "__main__":
    print(f"Token server http://{TOKEN_SERVER_HOST}:{TOKEN_SERVER_PORT}")
    if not TOKEN_SERVER_SECRET:
        print("Warning: TOKEN_SERVER_SECRET unset — /token is open on this host")
    app.run(host=TOKEN_SERVER_HOST, port=TOKEN_SERVER_PORT, debug=True)
