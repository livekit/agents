from typing import Callable, Dict, Set, Optional, Generic, TypeVar

T = TypeVar("T")


class EventEmitter(Generic[T]):
    def __init__(self) -> None:
        self._events: Dict[T, Set[Callable]] = dict()

    def emit(self, event: T, *args, **kwargs) -> None:
        if event in self._events:
            for callback in self._events[event]:
                callback(*args, **kwargs)

    def once(self, event: T, callback: Optional[Callable] = None) -> Callable:
        if callback is not None:

            def once_callback(*args, **kwargs):
                self.off(event, once_callback)
                callback(*args, **kwargs)

            return self.on(event, once_callback)
        else:

            def decorator(callback: Callable) -> Callable:
                self.once(event, callback)
                return callback

            return decorator

    def on(self, event: T, callback: Optional[Callable] = None) -> Callable:
        if callback is not None:
            if event not in self._events:
                self._events[event] = set()
            self._events[event].add(callback)
            return callback
        else:

            def decorator(callback: Callable) -> Callable:
                self.on(event, callback)
                return callback

            return decorator

    def off(self, event: T, callback: Callable) -> None:
        if event in self._events:
            self._events[event].remove(callback)
