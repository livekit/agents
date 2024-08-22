from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import multiprocessing.context as mpc
import multiprocessing.shared_memory as mp_shm
import socket
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable, Literal

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


@dataclass
class PaintData:
    dirty_rects: list[tuple[int, int, int, int]]
    frame: rtc.VideoFrame
    width: int
    height: int


@dataclass
class BrowserOptions:
    url: str
    framerate: int
    width: int
    height: int
    paint_callback: Callable[[PaintData], None]


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

        self._view_width = 0
        self._view_height = 0

        self._created_fut = asyncio.Future()
        self._close_fut = asyncio.Future()

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
        #  browser process crashed for some reasons)
        await asyncio.shield(self._created_fut)

    async def aclose(self) -> None:
        await ipc.channel.asend_message(
            self._ctx_duplex, proto.CloseBrowserRequest(page_id=self.id)
        )
        await asyncio.shield(self._close_fut)

        self._shm.unlink()
        self._shm.close()

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

        paint_data = PaintData(
            dirty_rects=acq.dirty_rects,
            frame=self._framebuffer,
            width=acq.width,
            height=acq.height,
        )
        self.emit("paint", paint_data)

        release_paint = proto.ReleasePaintData(page_id=acq.page_id)
        await ipc.channel.asend_message(self._ctx_duplex, release_paint)

    async def _handle_close(self, msg: proto.BrowserClosed) -> None:
        logger.debug("browser page closed", extra={"page_id": self.id})
        self._close_fut.set_result(None)


class BrowserContext:
    def __init__(self, *, dev_mode: bool, remote_debugging_port: int = 0) -> None:
        self._mp_ctx = mp.get_context("spawn")
        self._pages: dict[int, BrowserPage] = {}
        self._dev_mode = dev_mode
        self._initialized = False
        self._next_page_id = 1
        self._remote_debugging_port = remote_debugging_port

    async def initialize(self) -> None:
        mp_pch, mp_cch = socket.socketpair()
        self._duplex = await utils.aio.duplex_unix._AsyncDuplex.open(mp_pch)

        self._proc = self._mp_ctx.Process(target=proc_main.main, args=(mp_cch,))
        self._proc.start()
        mp_cch.close()

        if not self._remote_debugging_port:
            with contextlib.closing(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ) as s:
                s.bind(("", 0))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._remote_debugging_port = s.getsockname()[1]

            logger.debug("using remote debugging port %d", self._remote_debugging_port)

        await ipc.channel.asend_message(
            self._duplex,
            proto.InitializeContextRequest(
                dev_mode=self._dev_mode,
                remote_debugging_port=self._remote_debugging_port,
                root_cache_path=tempfile.mkdtemp(),  # TODO(theomonnom): cleanup
            ),
        )
        resp = await ipc.channel.arecv_message(self._duplex, proto.IPC_MESSAGES)
        assert isinstance(resp, proto.ContextInitializedResponse)
        self._initialized = True
        logger.debug("browser context initialized", extra={"pid": self._proc.pid})

        self._main_atask = asyncio.create_task(self._main_task(self._duplex))

    @asynccontextmanager
    async def playwright(self, timeout: float | None = None):
        if not self._initialized:
            raise RuntimeError("BrowserContext not initialized")

        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            url = f"http://localhost:{self._remote_debugging_port}"
            browser = await p.chromium.connect_over_cdp(url, timeout=timeout)
            try:
                yield browser
            finally:
                await browser.close()

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
            elif isinstance(msg, proto.BrowserClosed):
                page = self._pages[msg.page_id]
                await page._handle_close(msg)

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
