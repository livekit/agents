from __future__ import annotations

import asyncio
import json
from typing import Any, List, Dict, TYPE_CHECKING, NamedTuple

from realtime.types import Callback

if TYPE_CHECKING:
    from realtime.connection import Socket


class CallbackListener(NamedTuple):
    """A tuple with `event` and `callback` """
    event: str
    callback: Callback


class Channel:
    """
    `Channel` is an abstraction for a topic listener for an existing socket connection.
    Each Channel has its own topic and a list of event-callbacks that responds to messages.
    Should only be instantiated through `connection.Socket().set_channel(topic)`
    Topic-Channel has a 1-many relationship.
    """

    def __init__(self, socket: Socket, topic: str, params: Dict[str, Any] = {}) -> None:
        """
        :param socket: Socket object
        :param topic: Topic that it subscribes to on the realtime server
        :param params:
        """
        self.socket = socket
        self.params = params
        self.topic = topic
        self.listeners: List[CallbackListener] = []
        self.joined = False

    def join(self) -> Channel:
        """
        Wrapper for async def _join() to expose a non-async interface
        Essentially gets the only event loop and attempt joining a topic
        :return: Channel
        """
        loop = asyncio.get_event_loop()  # TODO: replace with get_running_loop
        loop.run_until_complete(self._join())
        return self

    async def _join(self) -> None:
        """
        Coroutine that attempts to join Phoenix Realtime server via a certain topic
        :return: None
        """
        join_req = dict(topic=self.topic, event="phx_join",
                        payload={}, ref=None)

        try:
            await self.socket.ws_connection.send(json.dumps(join_req))
        except Exception as e:
            print(str(e))  # TODO: better error propagation
            return

    def on(self, event: str, callback: Callback) -> Channel:
        """
        :param event: A specific event will have a specific callback
        :param callback: Callback that takes msg payload as its first argument
        :return: Channel
        """
        cl = CallbackListener(event=event, callback=callback)
        self.listeners.append(cl)
        return self

    def off(self, event: str) -> None:
        """
        :param event: Stop responding to a certain event
        :return: None
        """
        self.listeners = [
            callback for callback in self.listeners if callback.event != event]
