class SupervisedProc:
    def __init__(
        self,
        *,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Coroutine],
        job_shutdown_fnc: Callable[[JobContext], Coroutine],
        mp_ctx: SpawnContext | ForkServerContext,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        log_q = mp.Queue()
        mp_pch, mp_cch = mp_ctx.Pipe(duplex=True)
        self._loop = loop
        self._proc_args = ProcStartArgs(
            initialize_process_fnc=initialize_process_fnc,
            job_entrypoint_fnc=job_entrypoint_fnc,
            job_shutdown_fnc=job_shutdown_fnc,
            log_q=log_q,
            mp_cch=mp_cch,
            asyncio_debug=loop.get_debug(),
        )

        self._pch = channel.ProcChannel(
            conn=mp_pch, loop=self._loop, messages=IPC_MESSAGES
        )
        self._proc = mp_ctx.Process(target=_proc_main, args=(self._proc_args,))
        self._proc_join_fut = asyncio.Future()

    def start(self) -> None:
        self._proc.start()

        def _sync_run():
            self._proc.join()
            self._loop.call_soon_threadsafe(self._proc_join_fut.set_result, None)

        thread = threading.Thread(target=_sync_run)
        thread.start()

    async def initialize(self) -> None:
        await self._pch.asend(InitializeRequest())

        init_res = await self._pch.arecv()  # wait for the process to become ready
        assert isinstance(
            init_res, InitializeResponse
        ), "first message must be InitializeResponse"

        self._monitor_atask = asyncio.create_task(self._monitor_task())

    async def join(self) -> None:
        await self._proc_join_fut

    async def aclose(self) -> None:
        await self._pch.aclose()
        await self.join()
        self._proc.close()

    def kill(self) -> None:
        if not self._proc.is_alive():
            return

        logger.info("killing job process")
        if sys.platform == "win32":
            self._proc.terminate()
        else:
            self._proc.kill()

    async def _monitor_task(self) -> None:
        ping_interval = utils.aio.interval(PING_INTERVAL)
        pong_timeout = utils.aio.sleep(PING_TIMEOUT)

        async def _ping_co() -> None:
            while True:
                await ping_interval.tick()
                await self._pch.asend(PingRequest(timestamp=utils.time_ms()))

        async def _pong_co() -> None:
            await pong_timeout
            logger.error("job ping timeout, killing job", extra=self.logging_extra())
            self.kill()

        async def _read_ipc_co() -> None:
            while True:
                msg = await self._pch.arecv()
                print(msg)

        try:
            await asyncio.gather(_ping_co(), _read_ipc_co())
        except channel.ChannelClosed:
            pass

    def logging_extra(self) -> dict:
        return {"pid": self._proc.pid}
