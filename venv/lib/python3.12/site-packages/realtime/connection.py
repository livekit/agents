import asyncio
import json
import logging
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, List, Dict, TypeVar, DefaultDict

import websockets
from typing_extensions import ParamSpec

from realtime.channel import Channel
from realtime.exceptions import NotConnectedError
from realtime.message import HEARTBEAT_PAYLOAD, PHOENIX_CHANNEL, ChannelEvents, Message
from realtime.types import Callback, T_ParamSpec, T_Retval

logging.basicConfig(
    format="%(asctime)s:%(levelname)s - %(message)s", level=logging.INFO
)


def ensure_connection(func: Callback):
    @wraps(func)
    def wrapper(*args: T_ParamSpec.args, **kwargs: T_ParamSpec.kwargs) -> T_Retval:
        if not args[0].connected:
            raise NotConnectedError(func.__name__)

        return func(*args, **kwargs)

    return wrapper


class Socket:
    def __init__(
        self,
        url: str,
        auto_reconnect: bool = False,
        params: Dict[str, Any] = {},
        hb_interval: int = 5,
    ) -> None:
        """
        `Socket` is the abstraction for an actual socket connection that receives and 'reroutes' `Message` according to its `topic` and `event`.
        Socket-Channel has a 1-many relationship.
        Socket-Topic has a 1-many relationship.
        :param url: Websocket URL of the Realtime server. starts with `ws://` or `wss://`
        :param params: Optional parameters for connection.
        :param hb_interval: WS connection is kept alive by sending a heartbeat message. Optional, defaults to 5.
        """
        self.url = url
        self.channels = defaultdict(list)
        self.connected = False
        self.params = params
        self.hb_interval = hb_interval
        self.ws_connection: websockets.client.WebSocketClientProtocol
        self.kept_alive = False
        self.auto_reconnect = auto_reconnect

        self.channels: DefaultDict[str, List[Channel]] = defaultdict(list)

    @ensure_connection
    def listen(self) -> None:
        """
        Wrapper for async def _listen() to expose a non-async interface
        In most cases, this should be the last method executed as it starts an infinite listening loop.
        :return: None
        """
        loop = asyncio.get_event_loop()  # TODO: replace with get_running_loop
        loop.run_until_complete(asyncio.gather(self._listen(), self._keep_alive()))

    async def _listen(self) -> None:
        """
        An infinite loop that keeps listening.
        :return: None
        """
        while True:
            try:
                msg = await self.ws_connection.recv()
                msg = Message(**json.loads(msg))

                if msg.event == ChannelEvents.reply:
                    continue

                for channel in self.channels.get(msg.topic, []):
                    for cl in channel.listeners:
                        if cl.event in ["*", msg.event]:
                            cl.callback(msg.payload)
            except websockets.exceptions.ConnectionClosed:
                if self.auto_reconnect:
                    logging.info(
                        "Connection with server closed, trying to reconnect..."
                    )
                    await self._connect()
                    for topic, channels in self.channels.items():
                        for channel in channels:
                            await channel._join()
                else:
                    logging.exception("Connection with the server closed.")
                    break

    def connect(self) -> None:
        """
        Wrapper for async def _connect() to expose a non-async interface
        """
        loop = asyncio.get_event_loop()  # TODO: replace with get_running
        loop.run_until_complete(self._connect())
        self.connected = True

    async def _connect(self) -> None:
        ws_connection = await websockets.connect(self.url)

        if ws_connection.open:
            logging.info("Connection was successful")
            self.ws_connection = ws_connection
            self.connected = True
        else:
            raise Exception("Connection Failed")

    async def _keep_alive(self) -> None:
        """
        Sending heartbeat to server every 5 seconds
        Ping - pong messages to verify connection is alive
        """
        while True:
            try:
                data = dict(
                    topic=PHOENIX_CHANNEL,
                    event=ChannelEvents.heartbeat,
                    payload=HEARTBEAT_PAYLOAD,
                    ref=None,
                )
                await self.ws_connection.send(json.dumps(data))
                await asyncio.sleep(self.hb_interval)
            except websockets.exceptions.ConnectionClosed:
                if self.auto_reconnect:
                    logging.info(
                        "Connection with server closed, trying to reconnect..."
                    )
                    await self._connect()
                else:
                    logging.exception("Connection with the server closed.")
                    break

    @ensure_connection
    def set_channel(self, topic: str) -> Channel:
        """
        :param topic: Initializes a channel and creates a two-way association with the socket
        :return: Channel
        """
        chan = Channel(self, topic, self.params)
        self.channels[topic].append(chan)

        return chan

    def summary(self) -> None:
        """
        Prints a list of topics and event the socket is listening to
        :return: None
        """
        for topic, chans in self.channels.items():
            for chan in chans:
                print(f"Topic: {topic} | Events: {[e for e, _ in chan.callbacks]}]")
