from importlib import import_module
import asyncio
import threading
import queue
import logging
import ctypes

class Mp3StreamDecoder:
    def __init__(self):
        try:
            globals()["av"] = import_module("av")
        except ImportError:
            raise ImportError("You haven't included the decoder_utils optional dependencies. Please install the decoder_utils extra by running `pip install livekit-agents[decoder_utils]`")
        self._input_queue = queue.Queue()
        self._packet_queue = queue.Queue()
        self._output_queue = queue.Queue()
        self._closed = False
        self._codec = av.CodecContext.create('mp3', 'r') # noqa
        self._packet_parse_thread = threading.Thread(target=self._packet_parse_thread)
        self._packet_parse_thread.start()
        self._decode_thread = threading.Thread(target=self._decode_thread)
        self._decode_thread.start()

    def flush(self):
        self._input_queue.put(None, block=False)

    def close(self):
        self._closed = True
        self._output_queue.put(None, block=False)

    def push_chunk(self, chunk: bytes):
        if self._closed:
            raise ValueError("Cannot push chunk to closed decoder")
        self._input_queue.put(chunk, block=False)

    def _packet_parse_thread(self):
        while True:
            data = self._input_queue.get()
            if data is None:
                self._packet_queue.put(None, block=False)
                continue

            packets = self._codec.parse(data)
            for packet in packets:
                self._packet_queue.put(packet, block=False)

    def _decode_thread(self):
        while True:
            packet = self._packet_queue.get()
            if packet is None and self._closed:
                self._packet_queue.put(None, block=False)
                break
            try:
                decoded = self._codec.decode(packet)
                # flush case
                if packet is None:
                    print("creating new codec")
                    self._codec = av.CodecContext.create('mp3', 'r') # noqa
                    continue
                print(decoded)
                plane = decoded[0].planes[0]
                ptr = plane.buffer_ptr
                size = plane.buffer_size
                byte_array_pointer = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_char * size))
                bytes_object = bytes(byte_array_pointer.contents)
                self._output_queue.put(bytes_object, block=False)
            except Exception as e:
                logging.exception("Error decoding mp3 chunk", e)
                continue

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            try:
                packet = self._output_queue.get_nowait()
                if packet is None:
                    raise StopAsyncIteration
                return packet
            except queue.Empty:
                await asyncio.sleep(0.01)