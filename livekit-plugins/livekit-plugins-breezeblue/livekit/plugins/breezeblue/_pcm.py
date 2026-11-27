from struct import iter_unpack

BYTES_PER_SECOND = 48_000
_QUIET_PCM_LEVEL = 256


class PcmPause:
    """Top up the silence between independent 24 kHz PCM streams."""

    def __init__(self, milliseconds: int) -> None:
        self._limit = milliseconds * 48
        self._tail = b""
        self._head = b""
        self.padding: int | None = 0

    def begin(self) -> None:
        self._head = b""
        self.padding = None if self._tail else 0

    def _remember(self, pcm: bytes) -> None:
        if self._limit:
            self._tail = (self._tail + pcm)[-self._limit :]

    def push(self, pcm: bytes) -> tuple[bytes, bytes]:
        if self.padding is not None:
            self._remember(pcm)
            return b"", pcm

        head = self._head + pcm
        # Each value is one signed little-endian PCM16 sample.
        left = next(
            (
                i * 2
                for i, (sample,) in enumerate(iter_unpack("<h", head[: self._limit]))
                if abs(sample) >= _QUIET_PCM_LEVEL
            ),
            None,
        )
        if left is None and len(head) < self._limit:
            self._head = head
            return b"", b""
        right = next(
            (
                i * 2
                for i in range(len(self._tail) // 2)
                if abs(
                    int.from_bytes(
                        self._tail[-2 * (i + 1) : len(self._tail) - 2 * i], "little", signed=True
                    )
                )
                >= _QUIET_PCM_LEVEL
            ),
            len(self._tail),
        )
        self.padding = max(0, self._limit - right - (left if left is not None else self._limit))
        self._head = b""
        self._remember(head)
        return bytes(self.padding), head

    def finish(self) -> bytes:
        """Forward a short silent stream as provided, without padding it."""
        pcm, self._head = self._head, b""
        self.padding = self.padding or 0
        self._remember(pcm)
        return pcm
