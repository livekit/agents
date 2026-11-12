from os import PathLike
from pathlib import Path

import grpc

class Auth:
    ssl_cert: Path | None
    uri: str
    use_ssl: bool
    channel: grpc.Channel
    def __init__(
        self,
        ssl_cert: str | PathLike[str] | None = None,
        use_ssl: bool = False,
        uri: str = "localhost:50051",
        metadata_args: list[list[str]] | None = None,
    ) -> None: ...
    def get_auth_metadata(self) -> list[tuple[str, str]]: ...
