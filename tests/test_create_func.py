import enum
from inspect import _empty
from typing import Annotated, List, Optional

import pytest
from livekit.agents import llm
from livekit.plugins.openai import _oai_api


def test_func_basic():
    class TestFunctionContext(llm.FunctionContext):
        @llm.ai_callable(name="test_function", description="A simple test function")
        def test_fn(self, param: Annotated[str, llm.TypeInfo(description="A string parameter")]):
            pass

    fnc_ctx = TestFunctionContext()
    assert "test_function" in fnc_ctx.ai_functions, "Function should be registered in ai_functions"

    fnc_info = fnc_ctx.ai_functions["test_function"]
    build_info = _oai_api.build_oai_function_description(fnc_info)
    assert fnc_info.name == build_info["function"]["name"]
    assert fnc_info.description == build_info["function"]["description"]
    assert not fnc_info.auto_retry
    assert "param" in fnc_info.arguments
    assert "param" in build_info["function"]["parameters"]["properties"]
    assert "param" in build_info["function"]["parameters"]["required"]

    arg_info = fnc_info.arguments["param"]
    build_arg_info = build_info["function"]["parameters"]["properties"]["param"]

    assert arg_info.name == "param"
    assert arg_info.description == "A string parameter"
    assert arg_info.type is str
    assert arg_info.default is _empty
    assert arg_info.choices == ()
    assert build_arg_info["description"] == arg_info.description
    assert build_arg_info["type"] == "string"


def test_func_duplicate():
    class TestFunctionContext(llm.FunctionContext):
        @llm.ai_callable(name="duplicate_function", description="A simple test function")
        def fn1(self):
            pass

        @llm.ai_callable(name="duplicate_function", description="A simple test function")
        def fn2(self):
            pass

    with pytest.raises(ValueError, match="duplicate ai_callable name: duplicate_function"):
        TestFunctionContext()


def test_func_with_docstring():
    class TestFunctionContext(llm.FunctionContext):
        @llm.ai_callable()
        def test_fn(self):
            """A simple test function"""
            pass

    fnc_ctx = TestFunctionContext()
    assert "test_fn" in fnc_ctx.ai_functions, "Function should be registered in ai_functions"

    assert fnc_ctx.ai_functions["test_fn"].description == "A simple test function"


def test_func_with_optional_parameter():
    class TestFunctionContext(llm.FunctionContext):
        @llm.ai_callable(name="optional_function", description="Function with optional parameter")
        def optional_fn(
            self,
            param: Annotated[
                Optional[int], llm.TypeInfo(description="An optional integer parameter")
            ] = None,
            param2: Optional[List[str]] = None,
            param3: str = "A string",
        ):
            pass

    fnc_ctx = TestFunctionContext()
    assert "optional_function" in fnc_ctx.ai_functions, (
        "Function should be registered in ai_functions"
    )

    fnc_info = fnc_ctx.ai_functions["optional_function"]
    build_info = _oai_api.build_oai_function_description(fnc_info)
    print(build_info)
    assert fnc_info.name == build_info["function"]["name"]
    assert fnc_info.description == build_info["function"]["description"]
    assert "param" in fnc_info.arguments
    assert "param2" in fnc_info.arguments
    assert "param3" in fnc_info.arguments
    assert "param" in build_info["function"]["parameters"]["properties"]
    assert "param2" in build_info["function"]["parameters"]["properties"]
    assert "param3" in build_info["function"]["parameters"]["properties"]
    assert "param" not in build_info["function"]["parameters"]["required"]
    assert "param2" not in build_info["function"]["parameters"]["required"]
    assert "param3" not in build_info["function"]["parameters"]["required"]

    # Check 'param'
    arg_info = fnc_info.arguments["param"]
    build_arg_info = build_info["function"]["parameters"]["properties"]["param"]

    assert arg_info.name == "param"
    assert arg_info.description == "An optional integer parameter"
    assert arg_info.type == Optional[int]
    assert arg_info.default is None
    assert arg_info.choices == ()
    assert build_arg_info["description"] == arg_info.description
    assert build_arg_info["type"] == "number"

    # Check 'param2'
    arg_info = fnc_info.arguments["param2"]
    build_arg_info = build_info["function"]["parameters"]["properties"]["param2"]

    assert arg_info.name == "param2"
    assert arg_info.description == ""
    assert arg_info.type == Optional[List[str]]
    assert arg_info.default is None
    assert arg_info.choices == ()
    assert build_arg_info["type"] == "array"
    assert build_arg_info["items"]["type"] == "string"

    # check 'param3'
    arg_info = fnc_info.arguments["param3"]
    build_arg_info = build_info["function"]["parameters"]["properties"]["param3"]

    assert arg_info.name == "param3"
    assert arg_info.description == ""
    assert arg_info.type is str
    assert arg_info.default == "A string"
    assert arg_info.choices == ()
    assert build_arg_info["type"] == "string"


def test_func_with_list_parameter():
    class TestFunctionContext(llm.FunctionContext):
        @llm.ai_callable(name="list_function", description="Function with list parameter")
        def list_fn(
            self,
            items: Annotated[List[str], llm.TypeInfo(description="A list of strings")],
        ):
            pass

    fnc_ctx = TestFunctionContext()
    assert "list_function" in fnc_ctx.ai_functions, "Function should be registered in ai_functions"

    fnc_info = fnc_ctx.ai_functions["list_function"]
    build_info = _oai_api.build_oai_function_description(fnc_info)
    assert fnc_info.name == build_info["function"]["name"]
    assert fnc_info.description == build_info["function"]["description"]
    assert not fnc_info.auto_retry
    assert "items" in fnc_info.arguments
    assert "items" in build_info["function"]["parameters"]["properties"]
    assert "items" in build_info["function"]["parameters"]["required"]

    arg_info = fnc_info.arguments["items"]
    build_arg_info = build_info["function"]["parameters"]["properties"]["items"]

    assert arg_info.name == "items"
    assert arg_info.description == "A list of strings"
    assert arg_info.type is List[str]
    assert arg_info.default is _empty
    assert arg_info.choices == ()
    assert build_arg_info["description"] == arg_info.description
    assert build_arg_info["type"] == "array"
    assert build_arg_info["items"]["type"] == "string"


def test_func_with_enum_parameter():
    class Status(enum.Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"
        PENDING = "pending"

    class TestFunctionContext(llm.FunctionContext):
        @llm.ai_callable(name="enum_function", description="Function with enum parameter")
        def enum_fn(
            self,
            status: Annotated[Status, llm.TypeInfo(description="Status of the entity")],
        ):
            pass

    fnc_ctx = TestFunctionContext()
    assert "enum_function" in fnc_ctx.ai_functions, "Function should be registered in ai_functions"

    fnc_info = fnc_ctx.ai_functions["enum_function"]
    build_info = _oai_api.build_oai_function_description(fnc_info)
    assert fnc_info.name == build_info["function"]["name"]
    assert fnc_info.description == build_info["function"]["description"]
    assert not fnc_info.auto_retry
    assert "status" in fnc_info.arguments
    assert "status" in build_info["function"]["parameters"]["properties"]
    assert "status" in build_info["function"]["parameters"]["required"]

    arg_info = fnc_info.arguments["status"]
    build_arg_info = build_info["function"]["parameters"]["properties"]["status"]

    assert arg_info.name == "status"
    assert arg_info.description == "Status of the entity"
    assert arg_info.type is str  # Enum values are converted to their underlying type
    assert arg_info.default is _empty
    assert arg_info.choices == ("active", "inactive", "pending")
    assert build_arg_info["description"] == arg_info.description
    assert build_arg_info["type"] == "string"
    assert build_arg_info["enum"] == arg_info.choices
