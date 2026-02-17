from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar

from ..log import logger

K = TypeVar("K")
V = TypeVar("V")


class BoundedDict(OrderedDict[K, V]):
    def __init__(self, maxsize: int | None = None):
        super().__init__()
        self.maxsize = maxsize
        if self.maxsize is not None and self.maxsize <= 0:
            raise ValueError("maxsize must be greater than 0")

    def __setitem__(self, key: K, value: V) -> None:
        super().__setitem__(key, value)

        while self.maxsize is not None and len(self) > self.maxsize:
            self.popitem(last=False)

    def update_value(self, key: K, **kwargs: Any) -> V | None:
        """Update the value of a key with the given keyword arguments.
        Only update the value if the field value is not None and the field exists on the value.

        Args:
            key: The key to update.
            kwargs: The keyword arguments to update the value.

        Returns:
            The value of the key.
        """
        value = self.get(key, None)
        if value is None:
            return value
        for field_name, field_value in kwargs.items():
            if field_value is None:
                continue
            if hasattr(value, field_name):
                setattr(value, field_name, field_value)
            else:
                logger.warning(
                    "field %s is not set on value of type %s, skipping",
                    field_name,
                    type(value).__name__,
                )
        return value

    def set_or_update(self, key: K, factory: Callable[[], V], **kwargs: Any) -> V:
        """Set a value for a key if it doesn't exist, or update it if it does.

        Args:
            key: The key to set or update.
            factory: The factory function to create a new value if the key doesn't exist.
            kwargs: The keyword arguments to update the value.

        Returns:
            The value of the key.
        """
        if self.get(key, None) is None:
            self[key] = factory()
        result = self.update_value(key, **kwargs)
        assert result is not None
        return result

    def pop_if(
        self,
        predicate: Callable[[V], bool] | None = None,
    ) -> tuple[K | None, V | None]:
        """Pop an item from the dictionary if it satisfies the predicate.

        Args:
            predicate: The predicate to check if the value satisfies.

        Returns:
            A tuple of the key and value of the popped item.
        """
        if predicate is None:
            if len(self) > 0:
                return self.popitem(last=False)
            return None, None

        for key, value in reversed(list(self.items())):
            if predicate(value):
                return key, self.pop(key)
        return None, None
