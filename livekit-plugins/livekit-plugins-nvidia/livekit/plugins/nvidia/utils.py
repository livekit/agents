# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import grpc

from livekit.agents import APIConnectionError, APIError, APIStatusError, APITimeoutError


def to_api_error(error: Exception, *, operation: str) -> APIError:
    """Translate NVIDIA Speech client failures into LiveKit API errors."""
    if isinstance(error, APIError):
        return error

    if isinstance(error, grpc.RpcError):
        code = error.code()
        details = error.details() or str(error)
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return APITimeoutError(f"{operation} timed out: {details}")
        if code == grpc.StatusCode.CANCELLED:
            return APIConnectionError(f"{operation} was cancelled by NVIDIA Speech")

        status_codes = {
            grpc.StatusCode.INVALID_ARGUMENT: 400,
            grpc.StatusCode.NOT_FOUND: 404,
            grpc.StatusCode.ALREADY_EXISTS: 409,
            grpc.StatusCode.PERMISSION_DENIED: 403,
            grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
            grpc.StatusCode.FAILED_PRECONDITION: 400,
            grpc.StatusCode.ABORTED: 503,
            grpc.StatusCode.OUT_OF_RANGE: 400,
            grpc.StatusCode.UNIMPLEMENTED: 501,
            grpc.StatusCode.INTERNAL: 500,
            grpc.StatusCode.UNAVAILABLE: 503,
            grpc.StatusCode.DATA_LOSS: 500,
            grpc.StatusCode.UNAUTHENTICATED: 401,
        }
        return APIStatusError(
            f"{operation} failed: {details}",
            status_code=status_codes.get(code, -1),
            retryable=code
            in {
                grpc.StatusCode.UNKNOWN,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                grpc.StatusCode.ABORTED,
                grpc.StatusCode.INTERNAL,
                grpc.StatusCode.UNAVAILABLE,
            },
        )

    if isinstance(error, (TypeError, ValueError)):
        return APIStatusError(f"{operation} failed: {error}", status_code=400, retryable=False)

    return APIConnectionError(f"{operation} failed: {error}")
