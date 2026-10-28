from livekit.agents import APIConnectOptions, APIError


def request_timeout_ms(options: APIConnectOptions) -> int:
    # the SDK has no per-request way to disable a limit or to give phases different limits
    timeout = options.connect_timeout
    if (
        timeout is None
        or options.pool_timeout != timeout
        or options.read_timeout != timeout
        or options.write_timeout != timeout
        or not float(timeout * 1000).is_integer()
    ):
        raise APIError(
            "Mistral SDK applies one finite limit, in whole milliseconds, to every phase: "
            "pool/connect/write/read timeouts must be equal",
            retryable=False,
        )
    return int(timeout * 1000)
