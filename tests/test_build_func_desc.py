from inspect import _empty
from typing import Optional, Union

import pytest
from livekit.agents.llm import FunctionArgInfo, FunctionInfo
from livekit.agents.llm.function_context import _is_optional_type
from livekit.plugins.openai import _oai_api


def test_typing():
    assert _is_optional_type(Optional[int]) == (True, int)
    assert _is_optional_type(Union[str, None]) == (True, str)
    assert _is_optional_type(None | float) == (True, float)
    assert _is_optional_type(Union[str, int]) == (False, None)


@pytest.mark.parametrize(
    ("arg_typ", "oai_type"),
    [
        (int, "number"),
        (Optional[int], "number"),
        (None | int, "number"),
        (Union[None, int], "number"),
        (Union[str, None], "string"),
    ],
)
def test_description_building(arg_typ: type, oai_type: str):
    is_optional, inner_type = _is_optional_type(arg_typ)

    fi = FunctionInfo(
        name="foo",
        description="foo",
        auto_retry=False,
        callable=lambda: None,
        arguments={
            "arg": FunctionArgInfo(
                name="foo",
                description="foo",
                type=inner_type,
                is_optional=is_optional,
                default=_empty,
                choices=(),
            ),
        },
    )
    assert (
        _oai_api.build_oai_function_description(fi)["function"]["parameters"][
            "properties"
        ]["foo"]["type"]
        == oai_type
    )
