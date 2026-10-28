from livekit.agents import APIConnectOptions, APIError


def request_timeout_ms(options: APIConnectOptions) -> int:
    timeout = options.connect_timeout
    if (
        timeout is None
        or options.pool_timeout != timeout
        or options.read_timeout != timeout
        or options.write_timeout != timeout
        or options.total_timeout is not None
        or not float(timeout * 1000).is_integer()
    ):
        raise APIError(
            "Mistral SDK requires equal finite pool/connect/write/read timeouts in whole "
            "milliseconds and cannot enforce total_timeout across framework attempts",
            retryable=False,
        )
    return int(timeout * 1000)
