from __future__ import annotations

import aioboto3
import boto3
from botocore.exceptions import NoCredentialsError

from livekit.agents import llm
from livekit.agents.llm import FunctionTool

__all__ = ["to_fnc_ctx", "get_aws_async_session"]
DEFAULT_REGION = "us-east-1"


def get_aws_async_session(
    region: str | None = None,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> aioboto3.Session:
    _validate_aws_credentials(api_key, api_secret)
    session = aioboto3.Session(
        aws_access_key_id=api_key,
        aws_secret_access_key=api_secret,
        region_name=region or DEFAULT_REGION,
    )
    return session


def _validate_aws_credentials(
    api_key: str | None = None,
    api_secret: str | None = None,
) -> None:
    try:
        session = boto3.Session(aws_access_key_id=api_key, aws_secret_access_key=api_secret)
        creds = session.get_credentials()
        if not creds:
            raise ValueError("No credentials found")
    except (NoCredentialsError, Exception) as e:
        raise ValueError(f"Unable to locate valid AWS credentials: {str(e)}") from e


def to_fnc_ctx(fncs: list[FunctionTool]) -> list[dict]:
    return [_build_tool_spec(fnc) for fnc in fncs]


def _build_tool_spec(fnc: FunctionTool) -> dict:
    fnc = llm.utils.build_legacy_openai_schema(fnc, internally_tagged=True)
    return {
        "toolSpec": _strip_nones(
            {
                "name": fnc["name"],
                "description": fnc["description"] if fnc["description"] else None,
                "inputSchema": {"json": fnc["parameters"] if fnc["parameters"] else {}},
            }
        )
    }


def _strip_nones(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}
