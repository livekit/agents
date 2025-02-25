import sys
from inspect import _empty
from typing import List, Optional, Union

import pytest
from livekit.agents.llm import FunctionArgInfo, FunctionInfo
from livekit.agents.llm.function_context import _is_optional_type
from livekit.plugins.openai import _oai_api


def test_typing():
    assert _is_optional_type(Optional[int]) == (True, int)
    assert _is_optional_type(Union[str, None]) == (True, str)
    if sys.version_info >= (3, 10):
        assert _is_optional_type(float | None) == (True, float)
    assert _is_optional_type(Union[str, int]) == (False, None)


@pytest.mark.parametrize(
    ("arg_typ", "oai_type"),
    [
        pytest.param(int, "number", id="int"),
        pytest.param(Optional[int], "number", id="optional[int]"),
        pytest.param(Union[None, int], "number", id="union[none, int]"),
        pytest.param(Union[str, None], "string", id="union[str, none]"),
        pytest.param(List[int], "array", id="list[int]"),
        pytest.param(Optional[List[int]], "array", id="optional[list[int]]"),
    ],
)
def test_description_building(arg_typ: type, oai_type: str):
    fi = FunctionInfo(
        name="foo",
        description="foo",
        auto_retry=False,
        callable=lambda: None,
        arguments={
            "arg": FunctionArgInfo(
                name="foo",
                description="foo",
                type=arg_typ,
                default=_empty,
                choices=(),
            ),
        },
    )
    assert (
        _oai_api.build_oai_function_description(fi)["function"]["parameters"]["properties"]["foo"][
            "type"
        ]
        == oai_type
    )
