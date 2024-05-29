import asyncio
import io
import multiprocessing as mp
import struct
import time
from typing import ClassVar, Literal, Tuple

from attrs import define

from .. import apipe, ipc_enc

PlotType = Literal["vad_raw", "vad_smoothed", "vad_dur", "raw_t_vol", "vol"]
EventType = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
]


@define(kw_only=True)
class PlotMessage:
    MSG_ID: ClassVar[int] = 1

    which: PlotType = "vad_raw"
    x: float = 0.0
    y: float = 0.0

    def write(self, b: io.BytesIO) -> None:
        b.write(len(self.which).to_bytes(4, byteorder="big"))
        b.write(self.which.encode())
        b.write(struct.pack("d", self.x))
        b.write(struct.pack("d", self.y))

    def read(self, b: io.BytesIO) -> None:
        which_len = int.from_bytes(b.read(4), byteorder="big")
        self.which = b.read(which_len).decode()  # type: ignore
        self.x = struct.unpack("d", b.read(8))[0]
        self.y = struct.unpack("d", b.read(8))[0]


@define(kw_only=True)
class PlotEventMessage:
    MSG_ID: ClassVar[int] = 2

    which: EventType = "user_started_speaking"
    x: float = 0.0

    def write(self, b: io.BytesIO) -> None:
        b.write(len(self.which).to_bytes(4, byteorder="big"))
        b.write(self.which.encode())
        b.write(struct.pack("d", self.x))

    def read(self, b: io.BytesIO) -> None:
        which_len = int.from_bytes(b.read(4), byteorder="big")
        self.which = b.read(which_len).decode()  # type: ignore
        self.x = struct.unpack("d", b.read(8))[0]


PLT_MESSAGES: dict = {
    PlotMessage.MSG_ID: PlotMessage,
    PlotEventMessage.MSG_ID: PlotEventMessage,
}


def _draw_plot(reader: ipc_enc.ProcessPipeReader):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")
    mpl.rcParams["toolbar"] = "None"

    plot_data: dict[str, Tuple[list[float], list[float]]] = {}
    reader = reader

    fig, (pv, sp) = plt.subplots(2, sharex="all")
    fig.canvas.manager.set_window_title("Voice Assistant")  # type: ignore

    # not really accurate
    max_vad_points = 500
    max_vol_points = 1000

    def _draw_cb(sp, pv):
        nonlocal max_vol_points, max_vad_points
        while reader.poll():
            msg = ipc_enc.read_msg(reader, PLT_MESSAGES)
            if isinstance(msg, PlotMessage):
                data = plot_data.setdefault(msg.which, ([], []))
                data[0].append(msg.x)
                data[1].append(msg.y)

                max_points = (
                    max_vad_points if msg.which.startswith("vad") else max_vol_points
                )
                data[0][:] = data[0][-max_points:]
                data[1][:] = data[1][-max_points:]

        vad_raw = plot_data.setdefault("vad_raw", ([], []))
        vad_smoothed = plot_data.get("vad_smoothed", ([], []))
        # vad_dur = plot_data.get("vad_dur", ([], []))
        raw_t_vol = plot_data.get("raw_t_vol", ([], []))
        vol = plot_data.get("vol", ([], []))

        pv.clear()
        pv.set_ylim(0, 1)
        pv.set(ylabel="assistant volume")
        pv.plot(vol[0], vol[1], label="vol")
        pv.plot(raw_t_vol[0], raw_t_vol[1], label="raw")
        pv.legend()

        sp.clear()
        sp.set_ylim(0, 1)
        sp.set(xlabel="time (s)", ylabel="speech probability")
        sp.plot(vad_smoothed[0], vad_smoothed[1], label="prob")
        sp.plot(vad_raw[0], vad_raw[1], label="raw")
        sp.legend()

        # sd.clear()
        # sd.grid()
        # sd.set(xlabel="time (s)", ylabel="vad inference time (ms)")
        # sd.plot(vad_dur[0], vad_dur[1], label="inference_duration")
        # sd.legend()

        fig.canvas.draw()

    timer = fig.canvas.new_timer(interval=150)
    timer.add_callback(_draw_cb, sp, pv)
    timer.start()
    plt.show()


class AssistantPlotter:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._started = False

    def start(self):
        if self._started:
            return

        self._started = True
        self._start_time = time.time()
        pch, cch = mp.Pipe()
        self._plot_tx = apipe.AsyncPipe(pch, loop=self._loop, messages=PLT_MESSAGES)
        self._plot_proc = mp.Process(target=_draw_plot, args=(cch,), daemon=True)
        self._plot_proc.start()

    def plot_value(self, which: PlotType, y: float):
        if not self._started:
            return

        ts = time.time() - self._start_time
        asyncio.ensure_future(self._plot_tx.write(PlotMessage(which=which, x=ts, y=y)))

    def plot_event(self, which: EventType):
        if not self._started:
            return

        ts = time.time() - self._start_time
        asyncio.ensure_future(self._plot_tx.write(PlotEventMessage(which=which, x=ts)))

    def terminate(self):
        self._plot_proc.terminate()
