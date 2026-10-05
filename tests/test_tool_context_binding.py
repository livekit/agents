"""Regression tests for how ``@function_tool`` binds to an instance.

A plain function assigned as a class attribute also goes through
``_BaseFunctionTool.__get__``, but it has no receiver. Binding one anyway used to
delete its first parameter from the schema and then pass the instance as a
positional argument, so the model was never told the argument existed and the call
itself raised ``TypeError``.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from livekit.agents import function_tool
from livekit.agents.llm.tool_context import find_function_tools
from livekit.agents.llm.utils import function_arguments_to_pydantic_model

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def _schema_params(tool) -> list[str]:
    return sorted(function_arguments_to_pydantic_model(tool).model_json_schema()["properties"])


@function_tool
async def _plain_tool(order_id: str, region: str) -> str:
    """Look up an order."""
    return f"{order_id}/{region}"


class _PlainAttr:
    # A module-level function assigned as a class attribute. `self._plain_tool` reaches
    # `__get__`, but there is no receiver to bind.
    _plain_tool = _plain_tool


class TestFunctionToolInstanceBinding:
    def test_plain_function_class_attribute_keeps_its_first_parameter(self) -> None:
        bound = _PlainAttr()._plain_tool

        assert list(inspect.signature(bound).parameters) == ["order_id", "region"]
        assert _schema_params(bound) == ["order_id", "region"]

    def test_plain_function_class_attribute_is_callable(self) -> None:
        bound = _PlainAttr()._plain_tool

        assert asyncio.run(bound("A1", "eu")) == "A1/eu"

    def test_real_method_still_drops_the_receiver(self) -> None:
        """The method case must keep working: `self` is bound, not a schema parameter."""

        class Holder:
            @function_tool
            async def tool(self, order_id: str, region: str) -> str:  # noqa: ARG001
                """Look up an order."""
                return f"{order_id}/{region}"

        bound = Holder().tool

        assert list(inspect.signature(bound).parameters) == ["order_id", "region"]
        assert _schema_params(bound) == ["order_id", "region"]

    def test_real_method_is_callable(self) -> None:
        class Holder:
            @function_tool
            async def tool(self, order_id: str, region: str) -> str:  # noqa: ARG001
                """Look up an order."""
                return f"{order_id}/{region}"

        assert asyncio.run(Holder().tool("A1", "eu")) == "A1/eu"

    def test_return_annotation_is_preserved_for_a_real_method(self) -> None:
        """Dropping `self` must not disturb the rest of the signature."""

        class Holder:
            @function_tool
            async def tool(self, order_id: str) -> str:  # noqa: ARG001
                """Look up an order."""
                return order_id

        unbound = Holder.__dict__["tool"]._func  # type: ignore[attr-defined]
        bound_sig = inspect.signature(Holder().tool)

        assert bound_sig.return_annotation == inspect.signature(unbound).return_annotation

    def test_instance_and_class_discovery_agree_for_a_plain_function(self) -> None:
        """Both discovery paths must produce the same schema for the same declaration."""
        via_class = find_function_tools(_PlainAttr)
        via_instance = find_function_tools(_PlainAttr())

        assert [_schema_params(t) for t in via_class] == [_schema_params(t) for t in via_instance]
        assert _schema_params(via_instance[0]) == ["order_id", "region"]

    def test_inherited_method_still_binds(self) -> None:
        """A method inherited from a base class has `self` first and must still bind."""

        class Base:
            @function_tool
            async def tool(self, order_id: str) -> str:  # noqa: ARG001
                """Look up an order."""
                return order_id

        class Child(Base):
            pass

        assert list(inspect.signature(Child().tool).parameters) == ["order_id"]

    def test_find_function_tools_binds_custom_receiver_methods(self) -> None:
        class Holder:
            @function_tool
            async def tool(this, order_id: str) -> str:  # noqa: ARG001
                """Look up an order."""
                return order_id

        [tool] = find_function_tools(Holder())

        assert list(inspect.signature(tool).parameters) == ["order_id"]
        assert _schema_params(tool) == ["order_id"]
        assert asyncio.run(tool("A1")) == "A1"

    def test_find_function_tools_binds_inherited_custom_receiver_methods(self) -> None:
        class Base:
            @function_tool
            async def tool(this, order_id: str) -> str:  # noqa: ARG001
                """Look up an order."""
                return order_id

        class Child(Base):
            pass

        [tool] = find_function_tools(Child())

        assert list(inspect.signature(tool).parameters) == ["order_id"]
        assert _schema_params(tool) == ["order_id"]
        assert asyncio.run(tool("A1")) == "A1"

    def test_assigned_function_named_receiver_stays_a_tool_argument(self) -> None:
        @function_tool
        async def plain(this: str, order_id: str) -> str:
            """Look up an order."""
            return f"{this}/{order_id}"

        class Holder:
            tool = plain

        [tool] = find_function_tools(Holder())

        assert list(inspect.signature(tool).parameters) == ["this", "order_id"]
        assert _schema_params(tool) == ["order_id", "this"]
        assert asyncio.run(tool("source", "A1")) == "source/A1"

    def test_a_zero_argument_function_is_not_bound(self) -> None:
        @function_tool
        async def ping() -> str:
            """Ping."""
            return "pong"

        ping_tool = ping  # a class body cannot read a name it also assigns

        class Holder:
            ping = ping_tool

        assert list(inspect.signature(Holder().ping).parameters) == []
        assert asyncio.run(Holder().ping()) == "pong"
