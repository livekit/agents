import asyncio
import multiprocessing as mp
import multiprocessing.context as mpc
import multiprocessing.shared_memory as mp_shm
import socket
from dataclasses import dataclass
from typing import Literal

from livekit import rtc
from livekit.agents import ipc, utils

from . import logger, proc_main, proto


@dataclass
class _PageOptions:
    page_id: int
    url: str
    width: int
    height: int
    framerate: int


EventTypes = Literal["paint"]


class BrowserPage(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        mp_ctx: mpc.SpawnContext,
        opts: _PageOptions,
        ctx_duplex: utils.aio.duplex_unix._AsyncDuplex,
    ) -> None:
        super().__init__()
        self._mp_ctx = mp_ctx
        self._opts = opts
        self._ctx_duplex = ctx_duplex
        self._created_fut = asyncio.Future()

        self._view_width = 0
        self._view_height = 0

    @property
    def id(self) -> int:
        return self._opts.page_id

    async def start(self) -> None:
        shm_name = f"lkcef_browser_{utils.shortuuid()}"
        self._shm = mp_shm.SharedMemory(
            create=True,
            size=proto.SHM_MAX_WIDTH * proto.SHM_MAX_HEIGHT * 4,
            name=shm_name,
        )

        self._framebuffer = rtc.VideoFrame(
            proto.SHM_MAX_WIDTH,
            proto.SHM_MAX_HEIGHT,
            rtc.VideoBufferType.BGRA,
            bytearray(proto.SHM_MAX_WIDTH * proto.SHM_MAX_HEIGHT * 4),
        )

        req = proto.CreateBrowserRequest(
            page_id=self._opts.page_id,
            width=self._opts.width,
            height=self._opts.height,
            shm_name=shm_name,
            url=self._opts.url,
            framerate=self._opts.framerate,
        )

        await ipc.channel.asend_message(self._ctx_duplex, req)

        # TODO(theomonnom): create timeout (would prevent never resolving futures if the
        #   browser process crashed for some reasons)
        await asyncio.shield(self._created_fut)

    async def aclose(self) -> None:
        pass

    async def _handle_created(self, msg: proto.CreateBrowserResponse) -> None:
        self._created_fut.set_result(None)

    async def _handle_paint(self, acq: proto.AcquirePaintData) -> None:
        old_width = self._view_width
        old_height = self._view_height
        self._view_width = acq.width
        self._view_height = acq.height

        # TODO(theomonnom): remove hacky alloc-free resizing
        self._framebuffer._width = acq.width
        self._framebuffer._height = acq.height

        proto.copy_paint_data(
            acq, old_width, old_height, self._shm.buf, self._framebuffer.data
        )

        self.emit("paint", self._framebuffer)

        release_paint = proto.ReleasePaintData(page_id=acq.page_id)
        await ipc.channel.asend_message(self._ctx_duplex, release_paint)


class BrowserContext:
    def __init__(self, *, dev_mode: bool) -> None:
        self._mp_ctx = mp.get_context("spawn")
        self._pages: dict[int, BrowserPage] = {}
        self._dev_mode = dev_mode
        self._initialized = False
        self._next_page_id = 1

    async def initialize(self) -> None:
        mp_pch, mp_cch = socket.socketpair()
        self._duplex = await utils.aio.duplex_unix._AsyncDuplex.open(mp_pch)

        self._proc = self._mp_ctx.Process(target=proc_main.main, args=(mp_cch,))
        self._proc.start()
        mp_cch.close()

        await ipc.channel.asend_message(
            self._duplex, proto.InitializeContextRequest(dev_mode=self._dev_mode)
        )
        resp = await ipc.channel.arecv_message(self._duplex, proto.IPC_MESSAGES)
        assert isinstance(resp, proto.ContextInitializedResponse)
        self._initialized = True
        logger.debug("browser context initialized", extra={"pid": self._proc.pid})

        self._main_atask = asyncio.create_task(self._main_task(self._duplex))

    @utils.log_exceptions(logger)
    async def _main_task(self, duplex: utils.aio.duplex_unix._AsyncDuplex) -> None:
        while True:
            try:
                msg = await ipc.channel.arecv_message(duplex, proto.IPC_MESSAGES)
            except utils.aio.duplex_unix.DuplexClosed:
                break

            if isinstance(msg, proto.CreateBrowserResponse):
                page = self._pages[msg.page_id]
                await page._handle_created(msg)
            elif isinstance(msg, proto.AcquirePaintData):
                page = self._pages[msg.page_id]
                await page._handle_paint(msg)

    async def new_page(
        self, *, url: str, width: int = 800, height: int = 600, framerate: int = 30
    ) -> BrowserPage:
        if not self._initialized:
            raise RuntimeError("BrowserContext not initialized")

        page_id = self._next_page_id
        self._next_page_id += 1
        page = BrowserPage(
            self._mp_ctx,
            _PageOptions(
                page_id=page_id,
                url=url,
                width=width,
                height=height,
                framerate=framerate,
            ),
            self._duplex,
        )
        self._pages[page_id] = page
        await page.start()
        return page
