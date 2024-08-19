import asyncio
import socket
from livekit.agents import utils, ipc
import multiprocessing as mp
import multiprocessing.context as mpc
import multiprocessing.shared_memory as mp_shm

from . import proc_main, proto, logger


# there is no risk to increase these values. just using these defaults for now
# (they are used to calculate the size of the shared memory)
MAX_WIDTH = 1920
MAX_HEIGHT = 1080


# a page is in reality a browser in CEF
class BrowserPage:
    def __init__(self, mp_ctx: mpc.SpawnContext, page_id: int) -> None:
        self._mp_ctx = mp_ctx
        self._id = page_id
        self._created_fut = asyncio.Future()

    @property
    def id(self) -> int:
        return self._id

    async def start(self) -> None:
        shm_name = f"lkcef_browser_{utils.shortuuid()}"
        self._shm = mp_shm.SharedMemory(create=True, size=MAX_WIDTH * MAX_HEIGHT * 4, name=shm_name)

        # TODO(theomonnom): create timeout (would prevent never resolving futures if the
        # browser process crashed for some reasons)
        await asyncio.shield(self._created_fut)

    async def aclose(self) -> None:
        pass

    def _handle_created(self, msg: proto.CreateBrowserResponse) -> None:
        self._created_fut.set_result(None)

    def _handle_paint(self, msg: proto.AcquirePaintData) -> None:
        print("painting", msg.page_id, msg.width, msg.height)
        pass



class BrowserContext:
    def __init__(self, *, dev_mode: bool) -> None:
        self._mp_ctx = mp.get_context("spawn")
        self._pages: dict[int, BrowserPage] = {}
        self._dev_mode = dev_mode
        self._initialized = False
        self._next_page_id = 1

    async def initialize(self) -> None:
        mp_pch, mp_cch = socket.socketpair()
        duplex = await utils.aio.duplex_unix._AsyncDuplex.open(mp_pch)

        self._proc = self._mp_ctx.Process(target=proc_main.main, args=(mp_cch,))
        self._proc.start()
        mp_cch.close()

        await ipc.channel.asend_message(duplex, proto.InitializeContextRequest(dev_mode=self._dev_mode))
        resp = await ipc.channel.arecv_message(duplex, proto.IPC_MESSAGES)
        assert isinstance(resp, proto.ContextInitializedResponse)
        self._initialized = True
        logger.debug("browser context initialized", extra={"pid": self._proc.pid})


    async def _main_task(self, duplex: utils.aio.duplex_unix._AsyncDuplex) -> None:
        while True:
            try:
                msg = await ipc.channel.arecv_message(duplex, proto.IPC_MESSAGES)
            except utils.aio.duplex_unix.ChannelClosed:
                break

            if isinstance(msg, proto.CreateBrowserResponse):
                page = self._pages[msg.page_id]
                page._handle_created(msg)
            elif isinstance(msg, proto.AcquirePaintData):
                page = self._pages[msg.page_id]
                page._handle_paint(msg)

    async def new_page(self) -> BrowserPage:
        if not self._initialized:
            raise RuntimeError("BrowserContext not initialized")

        page_id = self._next_page_id
        self._next_page_id += 1

        page = BrowserPage(self._mp_ctx, page_id)
        self._pages[page_id] = page
        await page.start()
        return page
