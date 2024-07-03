from typing import Any, Callable, Dict, Generic, Optional, Set, TypeVar

T = TypeVar("T")


class EventEmitter(Generic[T]):
    def __init__(self) -> None:
        self._events: Dict[T, Set[Callable[..., Any]]] = dict()

    def emit(self, event: T, *args: Any, **kwargs: Any) -> None:
        if event in self._events:
            callables = self._events[event].copy()
            for callback in callables:
                callback(*args, **kwargs)

    def once(self, event: T, callback: Optional[Callable[..., Any]] = None):
        if callback is not None:

            def once_callback(*args: Any, **kwargs: Any):
                self.off(event, once_callback)
                callback(*args, **kwargs)

            return self.on(event, once_callback)
        else:

            def decorator(callback: Callable[..., Any]):
                self.once(event, callback)
                return callback

            return decorator

    def on(self, event: T, callback: Optional[Callable[..., Any]] = None):
        if callback is not None:
            if event not in self._events:
                self._events[event] = set()
            self._events[event].add(callback)
            return callback
        else:

            def decorator(callback: Callable[..., Any]):
                self.on(event, callback)
                return callback

            return decorator

    def off(self, event: T, callback: Callable[..., Any]) -> None:
        if event in self._events:
            self._events[event].remove(callback)
