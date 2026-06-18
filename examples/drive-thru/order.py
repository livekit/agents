from __future__ import annotations

import secrets
import string
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field


def order_uid() -> str:
    alphabet = string.ascii_uppercase + string.digits  # b36
    return "O_" + "".join(secrets.choice(alphabet) for _ in range(6))


class OrderedCombo(BaseModel):
    type: Literal["combo_meal"] = "combo_meal"
    order_id: str = Field(default_factory=order_uid)
    meal_id: str
    drink_id: str
    drink_size: Literal["M", "L"] | None
    fries_size: Literal["M", "L"]
    sauce_id: str | None


class OrderedHappy(BaseModel):
    type: Literal["happy_meal"] = "happy_meal"
    order_id: str = Field(default_factory=order_uid)
    meal_id: str
    drink_id: str
    drink_size: Literal["S", "M", "L"] | None
    sauce_id: str | None


class OrderedRegular(BaseModel):
    type: Literal["regular"] = "regular"
    order_id: str = Field(default_factory=order_uid)
    item_id: str
    size: Literal["S", "M", "L"] | None = None


OrderedItem = Annotated[OrderedCombo | OrderedHappy | OrderedRegular, Field(discriminator="type")]


@dataclass
class OrderState:
    items: dict[str, OrderedItem]

    async def add(self, item: OrderedItem) -> None:
        self.items[item.order_id] = item

    async def remove(self, order_id: str) -> OrderedItem:
        return self.items.pop(order_id)

    def get(self, order_id: str) -> OrderedItem | None:
        return self.items[order_id]
