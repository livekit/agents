from typing import ClassVar
from livekit.agents.ipc import channel
from dataclasses import dataclass, field

import io


@dataclass
class InitializeContextRequest:
    MSG_ID: ClassVar[int] = 0
    dev_mode: bool = False

    def write(self, b: io.BytesIO) -> None:
        channel.write_bool(b, self.dev_mode)

    def read(self, b: io.BytesIO) -> None:
        self.dev_mode = channel.read_bool(b)


@dataclass
class ContextInitializedResponse:
    MSG_ID: ClassVar[int] = 1


@dataclass
class CreateBrowserRequest:
    MSG_ID: ClassVar[int] = 2
    page_id: int = -1
    url: str = ""
    framerate: int = 0
    width: int = 0
    height: int = 0
    shm_name: str = ""

    def write(self, b: io.BytesIO) -> None:
        channel.write_int(b, self.page_id)
        channel.write_string(b, self.url)
        channel.write_int(b, self.framerate)
        channel.write_int(b, self.width)
        channel.write_int(b, self.height)
        channel.write_string(b, self.shm_name)

    def read(self, b: io.BytesIO) -> None:
        self.page_id = channel.read_int(b)
        self.url = channel.read_string(b)
        self.framerate = channel.read_int(b)
        self.width = channel.read_int(b)
        self.height = channel.read_int(b)
        self.shm_name = channel.read_string(b)


@dataclass
class CreateBrowserResponse:
    """
    This is going to wait for the created_callback to be called.
    (The create_browser function will be async)
    """

    MSG_ID: ClassVar[int] = 3
    page_id: int = -1
    browser_id: int = 0

    def write(self, b: io.BytesIO) -> None:
        channel.write_int(b, self.page_id)
        channel.write_int(b, self.browser_id)

    def read(self, b: io.BytesIO) -> None:
        self.page_id = channel.read_int(b)
        self.browser_id = channel.read_int(b)


@dataclass
class AcquirePaintData:
    MSG_ID: ClassVar[int] = 4
    page_id: int = -1
    width: int = 0
    height: int = 0
    dirty_rects: list[tuple[int, int, int, int]] = field(default_factory=list)

    def write(self, b: io.BytesIO) -> None:
        channel.write_int(b, self.page_id)
        channel.write_int(b, self.width)
        channel.write_int(b, self.height)
        channel.write_int(b, len(self.dirty_rects))
        for rect in self.dirty_rects:
            channel.write_int(b, rect[0])
            channel.write_int(b, rect[1])
            channel.write_int(b, rect[2])
            channel.write_int(b, rect[3])

    def read(self, b: io.BytesIO) -> None:
        self.page_id = channel.read_int(b)
        self.width = channel.read_int(b)
        self.height = channel.read_int(b)
        num_rects = channel.read_int(b)
        self.dirty_rects = []
        for _ in range(num_rects):
            x = channel.read_int(b)
            y = channel.read_int(b)
            width = channel.read_int(b)
            height = channel.read_int(b)
            self.dirty_rects.append((x, y, width, height))


@dataclass
class ReleasePaintData:
    MSG_ID: ClassVar[int] = 5
    page_id: int = -1

    def write(self, b: io.BytesIO) -> None:
        channel.write_int(b, self.page_id)

    def read(self, b: io.BytesIO) -> None:
        self.page_id = channel.read_int(b)


IPC_MESSAGES = {
    InitializeContextRequest.MSG_ID: InitializeContextRequest,
    ContextInitializedResponse.MSG_ID: ContextInitializedResponse,
    CreateBrowserRequest.MSG_ID: CreateBrowserRequest,
    CreateBrowserResponse.MSG_ID: CreateBrowserResponse,
    AcquirePaintData.MSG_ID: AcquirePaintData,
    ReleasePaintData.MSG_ID: ReleasePaintData,
}
