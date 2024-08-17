import asyncio
import contextlib
import io
import multiprocessing as mp
import selectors
import socket
import time
from dataclasses import dataclass
from typing import ClassVar, Literal, Tuple

from .. import utils
from ..ipc import channel

PlotType = Literal["vad_probability", "raw_vol", "smoothed_vol"]
EventType = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
]


@dataclass
class PlotMessage:
    MSG_ID: ClassVar[int] = 1

    which: PlotType = "vad_probability"
    x: float = 0.0
    y: float = 0.0

    def write(self, b: io.BytesIO) -> None:
        channel.write_string(b, self.which)
        channel.write_float(b, self.x)
        channel.write_float(b, self.y)

    def read(self, b: io.BytesIO) -> None:
        self.which = channel.read_string(b)  # type: ignore
        self.x = channel.read_float(b)
        self.y = channel.read_float(b)


@dataclass
class PlotEventMessage:
    MSG_ID: ClassVar[int] = 2

    which: EventType = "user_started_speaking"
    x: float = 0.0

    def write(self, b: io.BytesIO) -> None:
        channel.write_string(b, self.which)
        channel.write_float(b, self.x)

    def read(self, b: io.BytesIO) -> None:
        self.which = channel.read_string(b)  # type: ignore
        self.x = channel.read_float(b)


PLT_MESSAGES: dict = {
    PlotMessage.MSG_ID: PlotMessage,
    PlotEventMessage.MSG_ID: PlotEventMessage,
}


def _draw_plot(mp_cch):
    try:
        import matplotlib as mpl  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        raise ImportError(
            "matplotlib is required to run use the VoiceAssistant plotter"
        )

    plt.style.use("ggplot")
    mpl.rcParams["toolbar"] = "None"

    plot_data: dict[str, Tuple[list[float], list[float]]] = {}
    plot_events: dict[str, list[float]] = {}

    fig, (pv, sp) = plt.subplots(2, sharex="all")
    fig.canvas.manager.set_window_title("Voice Assistant")  # type: ignore

    max_points = 250

    duplex = utils.aio.duplex_unix._Duplex.open(mp_cch)

    selector = selectors.DefaultSelector()
    selector.register(mp_cch, selectors.EVENT_READ)

    def _draw_cb(sp, pv):
        while True:
            events = selector.select(timeout=0.01)
            if not events:
                break

            msg = channel.recv_message(duplex, PLT_MESSAGES)
            if isinstance(msg, PlotMessage):
                data = plot_data.setdefault(msg.which, ([], []))
                data[0].append(msg.x)
                data[1].append(msg.y)
                data[0][:] = data[0][-max_points:]
                data[1][:] = data[1][-max_points:]

                # remove old events older than 7.5s
                for events in plot_events.values():
                    while events and events[0] < msg.x - 7.5:
                        events.pop(0)

            elif isinstance(msg, PlotEventMessage):
                events = plot_events.setdefault(msg.which, [])
                events.append(msg.x)

        vad_raw = plot_data.setdefault("vad_probability", ([], []))
        raw_vol = plot_data.get("raw_vol", ([], []))
        vol = plot_data.get("smoothed_vol", ([], []))

        pv.clear()
        pv.set_ylim(0, 1)
        pv.set(ylabel="assistant volume")
        pv.plot(vol[0], vol[1], label="volume")
        pv.plot(raw_vol[0], raw_vol[1], label="target_volume")
        pv.legend()

        sp.clear()
        sp.set_ylim(0, 1)
        sp.set(xlabel="time (s)", ylabel="speech probability")
        sp.plot(vad_raw[0], vad_raw[1], label="raw")
        sp.legend()

        for start in plot_events.get("agent_started_speaking", []):
            pv.axvline(x=start, color="r", linestyle="--")

        for stop in plot_events.get("agent_stopped_speaking", []):
            pv.axvline(x=stop, color="r", linestyle="--")

        for start in plot_events.get("user_started_speaking", []):
            sp.axvline(x=start, color="r", linestyle="--")

        for stop in plot_events.get("user_stopped_speaking", []):
            sp.axvline(x=stop, color="r", linestyle="--")

        fig.canvas.draw()

    timer = fig.canvas.new_timer(interval=33)
    timer.add_callback(_draw_cb, sp, pv)
    timer.start()
    plt.show()


class AssistantPlotter:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._started = False

    async def start(self):
        if self._started:
            return

        mp_pch, mp_cch = socket.socketpair()
        self._duplex = await utils.aio.duplex_unix._AsyncDuplex.open(mp_pch)
        self._plot_proc = mp.Process(target=_draw_plot, args=(mp_cch,), daemon=True)
        self._plot_proc.start()
        mp_cch.close()

        self._started = True
        self._closed = False
        self._start_time = time.time()

    def plot_value(self, which: PlotType, y: float):
        if not self._started:
            return

        ts = time.time() - self._start_time
        self._send_message(PlotMessage(which=which, x=ts, y=y))

    def plot_event(self, which: EventType):
        if not self._started:
            return

        ts = time.time() - self._start_time
        self._send_message(PlotEventMessage(which=which, x=ts))

    def _send_message(self, msg: channel.Message) -> None:
        if self._closed:
            return

        async def _asend_message():
            try:
                await channel.asend_message(self._duplex, msg)
            except Exception:
                self._closed = True

        asyncio.ensure_future(_asend_message())

    async def terminate(self):
        if not self._started:
            return

        self._plot_proc.terminate()

        with contextlib.suppress(utils.aio.duplex_unix.DuplexClosed):
            await self._duplex.aclose()
