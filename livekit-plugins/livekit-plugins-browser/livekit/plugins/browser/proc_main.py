import importlib.resources
import multiprocessing.shared_memory as mp_shm
import socket
import threading

from livekit.agents import ipc, utils

from . import logger, proto


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

        self._closing = False
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
        opts.close_callback = bserver._closed
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
        if self._closing:
            return  # make sure to not use the shm

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

    def _closed(self) -> None:
        ipc.channel.send_message(
            self._duplex, proto.BrowserClosed(page_id=self._page_id)
        )

    def handle_release_paint(self, msg: proto.ReleasePaintData):
        self._release_paint_e.set()

    def handle_close(self, msg: proto.CloseBrowserRequest):
        self._closing = True
        self._impl.close()


def _manager_thread(duplex: utils.aio.duplex_unix._Duplex, browser_app):
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
        elif isinstance(msg, proto.CloseBrowserRequest):
            server = browsers[msg.page_id]
            server.handle_close(msg)
            del browsers[msg.page_id]


def main(mp_cch: socket.socket):
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
    opts.remote_debugging_port = init_req.remote_debugging_port
    opts.root_cache_path = init_req.root_cache_path
    opts.initialized_callback = _context_initialized

    res = (
        importlib.resources.files("livekit.plugins.browser.resources") / "lkcef_app.app"
    )
    with importlib.resources.as_file(res) as path:
        opts.framework_path = str(
            path / "Contents" / "Frameworks" / "Chromium Embedded Framework.framework"
        )
        opts.main_bundle_path = str(path)
        opts.subprocess_path = str(
            path
            / "Contents"
            / "Frameworks"
            / "lkcef Helper.app"
            / "Contents"
            / "MacOS"
            / "lkcef Helper"
        )

        app = lkcef.BrowserApp(opts)
        man_t = threading.Thread(target=_manager_thread, args=(duplex, app))
        man_t.start()

        app.run()  # run indefinitely
