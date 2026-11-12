"""RPC between room participants on an isolated local SFU."""

from __future__ import annotations

import uuid

import pytest

from livekit import rtc

from .sfu import connected


@pytest.mark.parametrize("response_timeout", [0.5, 0.01])
async def test_an_rpc_to_a_missing_method_fails_with_an_rpc_error_under_short_deadlines(
    sfu: str, response_timeout: float
) -> None:
    """An RPC to a method the peer never registered, with a deadline shorter than the SDK's
    connection timeout, raises an RpcError, and the connection keeps serving RPCs.
    https://github.com/livekit/agents/issues/2517"""
    name = "regression-" + uuid.uuid4().hex
    async with connected(sfu, name, "person") as person, connected(sfu, name, "agent") as agent:
        person.local_participant.register_rpc_method("echo", lambda data: data.payload)

        with pytest.raises(rtc.RpcError) as missing:
            await agent.local_participant.perform_rpc(
                destination_identity="person",
                method="chat_notification",
                payload="{}",
                response_timeout=response_timeout,
            )
        answer = await agent.local_participant.perform_rpc(
            destination_identity="person", method="echo", payload="still here"
        )

    assert missing.value.code in (
        rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD,
        rtc.RpcError.ErrorCode.RESPONSE_TIMEOUT,
        rtc.RpcError.ErrorCode.CONNECTION_TIMEOUT,
    )
    assert answer == "still here"
