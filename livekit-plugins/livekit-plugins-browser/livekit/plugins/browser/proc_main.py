import socket
import sys
import threading

from livekit.agents import utils, ipc
from . import proto, logger


def _manager_thread(duplex: utils.aio.duplex_unix._Duplex, browser_app):
    import lkcef_python as lkcef

    while True:
        try:
            msg = ipc.channel.recv_message(duplex, proto.IPC_MESSAGES)
        except utils.aio.duplex_unix.ChannedClosed:
            break

        if isinstance(msg, proto.CreateBrowserRequest):
            page_id = msg.page_id
            logger.debug(
                "creating browser",
                extra={
                    "page_id": page_id,
                    "url": msg.url,
                    "framerate": msg.framerate,
                    "width": msg.width,
                    "height": msg.height,
                    "shm_name": msg.shm_name,
                },
            )

            opts = lkcef.BrowserOptions()
            opts.framerate = msg.framerate
            opts.width = msg.width
            opts.height = msg.height

            def _browser_created(impl):
                browser_id = impl.identifier()
                logger.debug("browser created", extra={"browser_id": browser_id, "page_id": page_id})

                try:
                    ipc.channel.send_message(duplex,
                                             proto.CreateBrowserResponse(page_id=page_id, browser_id=browser_id))
                except utils.aio.duplex_unix.ChannedClosed:
                    logger.exception("failed to send CreateBrowserResponse")

            # this also returns impl (seems like I don't need it finally?)
            _ = browser_app.create_browser(msg.url,
                                           opts)


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
        except utils.aio.duplex_unix.ChannedClosed:
            logger.exception("failed to send ContextInitializedResponse")

    opts = lkcef.AppOptions()
    opts.dev_mode = init_req.dev_mode
    opts.initialized_callback = _context_initialized

    app = lkcef.BrowserApp(opts)
    man_t = threading.Thread(target=_manager_thread, args=(duplex, app))
    man_t.start()

    app.run()  # run indefinitely
