import socket
import sys
import threading

import multiprocessing.shared_memory as mp_shm

from livekit.agents import utils, ipc
from . import proto, logger


class BrowserServer:
    def __init__(
        self,
        duplex: utils.aio.duplex_unix._Duplex,
        shm: mp_shm.SharedMemory,
        page_id: int,
    ):
        self._duplex = duplex
        self._shm = shm
        self._page_id = page_id

        self._view_width = 0
        self._view_height = 0

        self._release_paint_e = threading.Event()

    @staticmethod
    def create(
        *,
        duplex: utils.aio.duplex_unix._Duplex,
        create_req: proto.CreateBrowserRequest,
        browser_app,
    ) -> "BrowserServer":
        logger.debug(
            "creating browser",
            extra={
                "page_id": create_req.page_id,
                "url": create_req.url,
                "framerate": create_req.framerate,
                "width": create_req.width,
                "height": create_req.height,
                "shm_name": create_req.shm_name,
            },
        )

        import lkcef_python as lkcef

        opts = lkcef.BrowserOptions()
        opts.framerate = create_req.framerate
        opts.width = create_req.width
        opts.height = create_req.height

        shm = mp_shm.SharedMemory(name=create_req.shm_name)
        bserver = BrowserServer(duplex, shm, create_req.page_id)

        opts.created_callback = bserver._browser_created
        opts.paint_callback = bserver._paint
        browser_app.create_browser(create_req.url, opts)
        return bserver

    def _browser_created(self, impl):
        browser_id = impl.identifier()
        logger.debug(
            "browser created",
            extra={"browser_id": browser_id, "page_id": self._page_id},
        )

        self._impl = impl

        try:
            ipc.channel.send_message(
                self._duplex,
                proto.CreateBrowserResponse(
                    page_id=self._page_id, browser_id=browser_id
                ),
            )
        except utils.aio.duplex_unix.DuplexClosed:
            logger.exception("failed to send CreateBrowserResponse")

    def _paint(self, frame_data):
        acq = proto.AcquirePaintData()
        acq.page_id = self._page_id
        acq.width = frame_data.width
        acq.height = frame_data.height

        dirty_rects = []
        for rect in frame_data.dirty_rects:
            dirty_rects.append((rect.x, rect.y, rect.width, rect.height))

        acq.dirty_rects = dirty_rects

        old_width = self._view_width
        old_height = self._view_height
        self._view_width = frame_data.width
        self._view_height = frame_data.height

        proto.copy_paint_data(
            acq, old_width, old_height, frame_data.buffer, self._shm.buf
        )

        try:
            ipc.channel.send_message(self._duplex, acq)
            self._release_paint_e.wait()  # wait for release
            self._release_paint_e.clear()
        except utils.aio.duplex_unix.DuplexClosed:
            logger.exception("failed to send AcquirePaintData")

    def handle_release_paint(self, msg: proto.ReleasePaintData):
        self._release_paint_e.set()


def _manager_thread(duplex: utils.aio.duplex_unix._Duplex, browser_app):
    import lkcef_python as lkcef

    browsers: dict[int, BrowserServer] = {}

    while True:
        try:
            msg = ipc.channel.recv_message(duplex, proto.IPC_MESSAGES)
        except utils.aio.duplex_unix.DuplexClosed:
            break

        if isinstance(msg, proto.CreateBrowserRequest):
            server = BrowserServer.create(
                duplex=duplex, create_req=msg, browser_app=browser_app
            )
            browsers[msg.page_id] = server
        elif isinstance(msg, proto.ReleasePaintData):
            server = browsers[msg.page_id]
            server.handle_release_paint(msg)


def main(mp_cch: socket.socket):
    sys.path.insert(
        0,
        "/Users/theomonnom/livekit/agents/livekit-plugins/livekit-plugins-browser/cef/src/Debug",
    )
    import lkcef_python as lkcef

    duplex = utils.aio.duplex_unix._Duplex.open(mp_cch)

    init_req = ipc.channel.recv_message(duplex, proto.IPC_MESSAGES)
    assert isinstance(init_req, proto.InitializeContextRequest)

    logger.debug("initializing browser context", extra={"dev_mode": init_req.dev_mode})

    def _context_initialized():
        try:
            ipc.channel.send_message(duplex, proto.ContextInitializedResponse())
        except utils.aio.duplex_unix.DuplexClosed:
            logger.exception("failed to send ContextInitializedResponse")

    opts = lkcef.AppOptions()
    opts.dev_mode = init_req.dev_mode
    opts.initialized_callback = _context_initialized

    app = lkcef.BrowserApp(opts)
    man_t = threading.Thread(target=_manager_thread, args=(duplex, app))
    man_t.start()

    app.run()  # run indefinitely
