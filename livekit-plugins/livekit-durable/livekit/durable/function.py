from __future__ import annotations

import functools
import os
from collections.abc import Coroutine, Generator
from types import (
    AsyncGeneratorType,
    CodeType,
    CoroutineType,
    FrameType,
    FunctionType,
    GeneratorType,
    MethodType,
    TracebackType,
)
from typing import Any, Callable, TypeVar, cast

import lk_durable as ext

from .registry import (
    RegisteredFunction,
    lookup_function,
    register_function,
    unregister_function,
)

TRACE = os.getenv("DISPATCH_TRACE", False)

FRAME_CLEARED = 4


class DurableFunction:
    """A wrapper for generator functions and async functions that make
    their generator and coroutine instances serializable."""

    # __slots__ = ("registered_fn", "__name__", "__qualname__")

    def __init__(self, fn: FunctionType):
        self.registered_fn = register_function(fn)
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        result = self.registered_fn.fn(*args, **kwargs)

        if isinstance(result, GeneratorType):
            return DurableGenerator(result, self.registered_fn, None, *args, **kwargs)
        elif isinstance(result, CoroutineType):
            return DurableCoroutine(result, self.registered_fn, *args, **kwargs)
        elif isinstance(result, AsyncGeneratorType):
            raise NotImplementedError(
                "only synchronous generator functions are supported at this time"
            )
        else:
            return result

    def __repr__(self) -> str:
        return f"DurableFunction({self.__qualname__})"

    def unregister(self):
        unregister_function(self.registered_fn.key)


def durable(fn: Callable) -> Callable:
    """Returns a "durable" function that creates serializable
    generators or coroutines."""
    if isinstance(fn, MethodType):
        static_fn = cast(FunctionType, fn.__func__)
        return MethodType(DurableFunction(static_fn), fn.__self__)
    elif isinstance(fn, FunctionType):
        return DurableFunction(fn)
    else:
        raise TypeError(f"cannot create a durable function from value of type {fn.__qualname__}")


class Serializable:
    """A wrapper for a generator or coroutine that makes it serializable."""

    __slots__ = (
        "g",
        "registered_fn",
        "wrapped_coroutine",
        "args",
        "kwargs",
        "__name__",
        "__qualname__",
    )

    g: GeneratorType | CoroutineType
    registered_fn: RegisteredFunction
    wrapped_coroutine: DurableCoroutine | None
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(
        self,
        g: GeneratorType | CoroutineType,
        registered_fn: RegisteredFunction,
        wrapped_coroutine: DurableCoroutine | None,
        *args: Any,
        **kwargs: Any,
    ):
        self.g = g
        self.registered_fn = registered_fn
        self.wrapped_coroutine = wrapped_coroutine
        self.args = args
        self.kwargs = kwargs
        self.__name__ = registered_fn.fn.__name__
        self.__qualname__ = registered_fn.fn.__qualname__

    def __getstate__(self):
        g = self.g
        rfn = self.registered_fn

        if g is None:
            frame_state = FRAME_CLEARED
        else:
            frame_state = ext.get_frame_state(g)

        if frame_state < FRAME_CLEARED:
            ip = ext.get_frame_ip(g)
            sp = ext.get_frame_sp(g)
            bp = ext.get_frame_bp(g)
            stack = [ext.get_frame_stack_at(g, i) for i in range(sp)]
            blocks = [ext.get_frame_block_at(g, i) for i in range(bp)]
        else:
            ip, sp, bp, stack, blocks = None, None, None, None, None

        if TRACE:
            print(f"\n[DISPATCH] Serializing {self}:")
            print(f"function = {rfn.fn.__qualname__} ({rfn.filename}:{rfn.lineno})")
            print(f"code hash = {rfn.hash}")
            print(f"args = {self.args}")
            print(f"kwargs = {self.kwargs}")
            print(f"wrapped coroutine = {self.wrapped_coroutine}")
            print(f"frame state = {frame_state}")
            if frame_state < FRAME_CLEARED:
                print(f"IP = {ip}")
                print(f"SP = {sp}")
                for i, (is_null, value) in enumerate(stack if stack is not None else []):
                    if is_null:
                        print(f"stack[{i}] = NULL")
                    else:
                        print(f"stack[{i}] = {value}")
                print(f"BP = {bp}")
                for i, block in enumerate(blocks if blocks is not None else []):
                    print(f"block[{i}] = {block}")
            print()

        state = {
            "function": {
                "key": rfn.key,
                "filename": rfn.filename,
                "lineno": rfn.lineno,
                "hash": rfn.hash,
                "args": self.args,
                "kwargs": self.kwargs,
            },
            "wrapped_coroutine": self.wrapped_coroutine,
            "frame": {
                "ip": ip,
                "sp": sp,
                "bp": bp,
                "stack": stack,
                "blocks": blocks,
                "state": frame_state,
            },
        }
        return state

    def __setstate__(self, state):
        function_state = state["function"]
        frame_state = state["frame"]

        # Recreate the generator/coroutine by looking up the constructor
        # and calling it with the same args/kwargs.
        key, filename, lineno, code_hash, args, kwargs = (
            function_state["key"],
            function_state["filename"],
            function_state["lineno"],
            function_state["hash"],
            function_state["args"],
            function_state["kwargs"],
        )
        wrapped_coroutine = state["wrapped_coroutine"]

        rfn = lookup_function(key)
        if filename != rfn.filename or lineno != rfn.lineno:
            raise ValueError(
                f"location mismatch for function {key}: {filename}:{lineno} vs. expected {rfn.filename}:{rfn.lineno}"
            )
        elif code_hash != rfn.hash:
            raise ValueError(
                f"hash mismatch for function {key}: {code_hash} vs. expected {rfn.hash}"
            )

        if frame_state["state"] < FRAME_CLEARED:
            if wrapped_coroutine:
                g = wrapped_coroutine.coroutine.__await__()
            else:
                g = rfn.fn(*args, **kwargs)

            # Restore the frame.
            ext.set_frame_ip(g, frame_state["ip"])
            ext.set_frame_sp(g, frame_state["sp"])
            for i, (is_null, obj) in enumerate(frame_state["stack"]):
                ext.set_frame_stack_at(g, i, is_null, obj)
            ext.set_frame_bp(g, frame_state["bp"])
            for i, block in enumerate(frame_state["blocks"]):
                ext.set_frame_block_at(g, i, block)
            ext.set_frame_state(g, frame_state["state"])
        else:
            g = None

        self.g = g
        self.registered_fn = rfn
        self.wrapped_coroutine = wrapped_coroutine
        self.args = args
        self.kwargs = kwargs

        self.__name__ = rfn.fn.__name__
        self.__qualname__ = rfn.fn.__qualname__


_YieldT = TypeVar("_YieldT", covariant=True)
_SendT = TypeVar("_SendT", contravariant=True)
_ReturnT = TypeVar("_ReturnT", covariant=True)


class DurableCoroutine(Serializable, Coroutine[_YieldT, _SendT, _ReturnT]):
    """A wrapper for a coroutine that makes it serializable (can be pickled).
    Instances behave like the coroutines they wrap."""

    __slots__ = ("coroutine",)

    def __init__(
        self,
        coroutine: CoroutineType,
        registered_fn: RegisteredFunction,
        *args: Any,
        **kwargs: Any,
    ):
        self.coroutine = coroutine
        Serializable.__init__(self, coroutine, registered_fn, None, *args, **kwargs)

    def __await__(self) -> Generator[Any, None, _ReturnT]:
        coroutine_wrapper = self.coroutine.__await__()
        generator = cast(GeneratorType, coroutine_wrapper)
        durable_coroutine_wrapper: Generator[Any, None, _ReturnT] = DurableGenerator(
            generator, self.registered_fn, self, *self.args, **self.kwargs
        )
        return durable_coroutine_wrapper

    def send(self, send: _SendT) -> _YieldT:
        return self.coroutine.send(send)

    def throw(
        self,
        typ: type[BaseException],
        val: BaseException | object = None,
        tb: TracebackType | None = None,
    ) -> _YieldT:
        return self.coroutine.throw(typ, val, tb)

    def close(self) -> None:
        self.coroutine.close()

    def __setstate__(self, state):
        Serializable.__setstate__(self, state)
        self.coroutine = cast(CoroutineType, self.g)

    @property
    def cr_running(self) -> bool:
        return self.coroutine.cr_running

    @property
    def cr_suspended(self) -> bool:
        return getattr(self.coroutine, "cr_suspended", False)

    @property
    def cr_code(self) -> CodeType:
        return self.coroutine.cr_code

    @property
    def cr_frame(self) -> FrameType:
        return self.coroutine.cr_frame

    @property
    def cr_await(self) -> Any:
        return self.coroutine.cr_await

    @property
    def cr_origin(self) -> tuple[tuple[str, int, str], ...] | None:
        return self.coroutine.cr_origin

    def __repr__(self) -> str:
        return f"DurableCoroutine({self.__qualname__})"


class DurableGenerator(Serializable, Generator[_YieldT, _SendT, _ReturnT]):
    """A wrapper for a generator that makes it serializable (can be pickled).
    Instances behave like the generators they wrap."""

    __slots__ = ("generator",)

    def __init__(
        self,
        generator: GeneratorType,
        registered_fn: RegisteredFunction,
        coroutine: DurableCoroutine | None,
        *args: Any,
        **kwargs: Any,
    ):
        self.generator = generator
        Serializable.__init__(self, generator, registered_fn, coroutine, *args, **kwargs)

    def __iter__(self) -> Generator[_YieldT, _SendT, _ReturnT]:
        return self

    def __next__(self) -> _YieldT:
        return next(self.generator)

    def send(self, send: _SendT) -> _YieldT:
        return self.generator.send(send)

    def throw(
        self,
        typ: type[BaseException],
        val: BaseException | object = None,
        tb: TracebackType | None = None,
    ) -> _YieldT:
        return self.generator.throw(typ, val, tb)

    def close(self) -> None:
        self.generator.close()

    def __setstate__(self, state):
        Serializable.__setstate__(self, state)
        self.generator = cast(GeneratorType, self.g)

    @property
    def gi_running(self) -> bool:
        return self.generator.gi_running

    @property
    def gi_suspended(self) -> bool:
        return getattr(self.generator, "gi_suspended", False)

    @property
    def gi_code(self) -> CodeType:
        return self.generator.gi_code

    @property
    def gi_frame(self) -> FrameType:
        return self.generator.gi_frame

    @property
    def gi_yieldfrom(self) -> GeneratorType | None:
        return self.generator.gi_yieldfrom

    def __repr__(self) -> str:
        if self.wrapped_coroutine is not None:
            return f"DurableCoroutineWrapper({self.__qualname__})"
        return f"DurableGenerator({self.__qualname__})"
