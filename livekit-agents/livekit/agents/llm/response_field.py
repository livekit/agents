from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel


class ResponseField:
    """Annotation marker for the field in an llm_output_format model that contains
    the spoken response text routed to TTS."""

    pass


Response = Annotated[str, ResponseField()]
"""Convenience type alias: ``response: Response`` is equivalent to
``response: Annotated[str, ResponseField()]``."""


def find_response_field(model: type[BaseModel]) -> str:
    """Scan a Pydantic model for the field annotated with ``ResponseField``.

    Returns the field name.

    Raises:
        ValueError: If zero or more than one field is annotated with ``ResponseField``.
    """
    found: list[str] = []
    for name, field_info in model.model_fields.items():
        if any(isinstance(m, ResponseField) for m in field_info.metadata):
            found.append(name)

    if len(found) == 0:
        raise ValueError(
            f"Model {model.__name__} has no field annotated with ResponseField. "
            f"Mark the spoken response field with `llm.Response` or "
            f"`Annotated[str, ResponseField()]`."
        )
    if len(found) > 1:
        raise ValueError(
            f"Model {model.__name__} has multiple fields annotated with ResponseField: "
            f"{found}. Only one field should be the spoken response."
        )
    return found[0]
