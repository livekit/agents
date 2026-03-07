import hashlib
from dataclasses import dataclass
from types import FunctionType
from typing import Any


@dataclass
class RegisteredFunction:
    """A function that can be referenced in durable state."""

    fn: FunctionType
    key: str
    filename: str
    lineno: int
    hash: str

    def __getstate__(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "filename": self.filename,
            "lineno": self.lineno,
            "hash": self.hash,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        key, filename, lineno, code_hash = (
            state["key"],
            state["filename"],
            state["lineno"],
            state["hash"],
        )

        rfn = lookup_function(key)
        if filename != rfn.filename or lineno != rfn.lineno:
            raise ValueError(
                f"location mismatch for function {key}: {filename}:{lineno} vs. expected {rfn.filename}:{rfn.lineno}"
            )
        elif code_hash != rfn.hash:
            raise ValueError(
                f"hash mismatch for function {key}: {code_hash} vs. expected {rfn.hash}"
            )

        # mypy 1.10.0 seems to report a false positive here:
        # error: Incompatible types in assignment (expression has type "FunctionType", variable has type "MethodType")  [assignment]
        self.fn = rfn.fn
        self.key = key
        self.filename = filename
        self.lineno = lineno
        self.hash = code_hash


_REGISTRY: dict[str, RegisteredFunction] = {}


def register_function(fn: FunctionType) -> RegisteredFunction:
    """Register a function in the in-memory function registry.

    When serializing a registered function, a reference to the function
    is stored along with details about its location and contents. When
    deserializing the function, the registry is consulted in order to
    find the function associated with the reference (and in order to
    check whether the function is the same).

    Args:
        fn: The function to register.

    Returns:
        str: Unique identifier for the function.

    Raises:
        ValueError: The function conflicts with another registered function.
    """
    rfn = RegisteredFunction(
        key=fn.__qualname__,
        fn=fn,
        filename=fn.__code__.co_filename,
        lineno=fn.__code__.co_firstlineno,
        hash="sha256:" + hashlib.sha256(fn.__code__.co_code).hexdigest(),
    )

    try:
        existing = _REGISTRY[rfn.key]
    except KeyError:
        pass
    else:
        if existing == rfn:
            return existing
        raise ValueError(f"durable function already registered with key {rfn.key}")

    _REGISTRY[rfn.key] = rfn
    return rfn


def lookup_function(key: str) -> RegisteredFunction:
    """Lookup a registered function by key.

    Args:
        key: Unique identifier for the function.

    Returns:
        RegisteredFunction: the function that was registered with the specified key.

    Raises:
        KeyError: A function has not been registered with this key.
    """
    return _REGISTRY[key]


def unregister_function(key: str) -> None:
    """Unregister a function by key.

    Args:
        key: Unique identifier for the function.

    Raises:
        KeyError: A function has not been registered with this key.
    """
    del _REGISTRY[key]


def clear_functions() -> None:
    """Clear functions clears the registry."""
    _REGISTRY.clear()
