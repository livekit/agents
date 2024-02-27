from importlib import import_module
import asyncio
import ctypes

class Mp3StreamDecoder:
    def __init__(self):
        try:
            globals()["av"] = import_module("av")
        except ImportError:
            raise ImportError("You haven't included the decoder_utils optional dependencies. Please install the decoder_utils extra by running `pip install livekit-agents[decoder_utils]`")
        self._closed = False
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()
        self._codec = av.CodecContext.create('mp3', 'r') # noqa
        self._run_task = asyncio.create_task(self._run())

    def close(self):
        self._closed = True
        self._input_queue.put_nowait(None)

    def push_chunk(self, chunk: bytes):
        if self._closed:
            raise ValueError("Cannot push chunk to closed decoder")
        self._input_queue.put_nowait(chunk)

    async def _run(self):
        while True:
            input = self._input_queue.get()
            if input is None:
                break

            result = await asyncio.to_thread(self._decode_input, input)
            self._output_queue.put_nowait(result)

    def _decode_input(self, input: bytes):
        packets = self._codec.parse(input)
        for packet in packets:
            decoded = self._codec.decode(packet)
            plane = decoded[0].planes[0]
            ptr = plane.buffer_ptr
            size = plane.buffer_size
            byte_array_pointer = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_char * size))
            return bytes(byte_array_pointer.contents)

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            packet = await self._output_queue.get()
            if packet is None:
                raise StopAsyncIteration
            return packet
