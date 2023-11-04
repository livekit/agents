# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from enum import Enum
from dataclasses import dataclass
from typing import Callable, TypeVar, Generic, AsyncIterable, Optional
from abc import abstractmethod

T = TypeVar('T')
U = TypeVar('U')

ProcessorEventType = Enum('ProcessorEventType', ['ERROR', 'SUCCESS'])


class Processor(Generic[T, U]):

    @dataclass
    class Event:
        type: ProcessorEventType
        data: Optional[U] = None
        error: Optional[Exception] = None

    def __init__(self, process: Callable[[AsyncIterable[T]], AsyncIterable[Event]]) -> None:
        self._process = process

    def start(self, data: AsyncIterable[T]) -> AsyncIterable[U]:
        return self._process(data)
