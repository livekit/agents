import multiprocessing

if multiprocessing.current_process().name == "job_proc":
    import signal
    import sys

    # ignore signals in the jobs process (the parent process will handle them)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    def _no_traceback_excepthook(exc_type, exc_val, traceback):
        if isinstance(exc_val, KeyboardInterrupt):
            return
        sys.__excepthook__(exc_type, exc_val, traceback)

    sys.excepthook = _no_traceback_excepthook


def proc_main(args) -> None:
    """main function for the job process when using the ProcessJobRunner"""

    # import every package lazily
    import asyncio
    import logging

    from .. import utils
    from ..job import JobProcess
    from ..log import logger
    from . import channel, job_main, proto

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)

    log_cch = utils.aio.duplex_unix._Duplex.open(args.log_cch)
    log_handler = job_main.LogQueueHandler(log_cch)
    root_logger.addHandler(log_handler)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_debug(args.asyncio_debug)
    loop.slow_callback_duration = 0.1  # 100ms
    utils.aio.debug.hook_slow_callbacks(2.0)

    cch = utils.aio.duplex_unix._Duplex.open(args.mp_cch)
    try:
        init_req = channel.recv_message(cch, proto.IPC_MESSAGES)

        assert isinstance(
            init_req, proto.InitializeRequest
        ), "first message must be InitializeRequest"

        job_proc = JobProcess(start_arguments=args.user_arguments)
        logger.info("initializing process", extra={"pid": job_proc.pid})
        args.initialize_process_fnc(job_proc)
        logger.info("process initialized", extra={"pid": job_proc.pid})
        channel.send_message(cch, proto.InitializeResponse())

        main_task = loop.create_task(
            job_main._async_main(job_proc, args.job_entrypoint_fnc, cch.detach()),
            name="job_proc_main",
        )
        while not main_task.done():
            try:
                loop.run_until_complete(main_task)
            except KeyboardInterrupt:
                # ignore the keyboard interrupt, we handle the process shutdown ourselves on the worker process
                pass
    except (utils.aio.duplex_unix.DuplexClosed, KeyboardInterrupt):
        pass
    finally:
        log_handler.close()
        loop.run_until_complete(loop.shutdown_default_executor())
