# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Oracle Corporation and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module is the Oracle LiveKit LLM plug-in.

Author: Keith Schnable (at Oracle Corporation)
Date: 2025-08-12
"""

from __future__ import annotations

import ast
import json

from livekit.agents import llm, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins.openai.utils import to_fnc_ctx

from .log import logger
from .oracle_llm import (
    CONTENT_TYPE_STRING,
    TOOL_CALL_DESCRIPTION,
    TOOL_CALL_PREFIX,
    BackEnd,
    OracleLLM,
    OracleLLMContent,
    OracleTool,
    OracleValue,
    Role,
)
from .utils import AuthenticationType


class LLM(llm.LLM):
    """
    The Oracle LiveKit LLM plug-in class. This derives from livekit.agents.llm.LLM.
    """

    def __init__(
        self,
        *,
        base_url: str,  # must be specified
        authentication_type: AuthenticationType = AuthenticationType.SECURITY_TOKEN,
        authentication_configuration_file_spec: str = "~/.oci/config",
        authentication_profile_name: str = "DEFAULT",
        back_end: BackEnd = BackEnd.GEN_AI_LLM,
        # these apply only if back_end == BackEnd.GEN_AI_LLM
        compartment_id: str | None = None,  # must be specified
        model_type: str = "GENERIC",  # must be "GENERIC" or "COHERE"
        model_id: str | None = None,  # must be specified or model_name must be specified
        model_name: str | None = None,  # must be specified or model_id must be specified
        maximum_number_of_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        # these apply only if back_end == BackEnd.GEN_AI_AGENT
        agent_endpoint_id: str | None = None,  # must be specified
    ) -> None:
        """
        Create a new instance of the OracleLLM class to access Oracle's GenAI service. This has LiveKit dependencies.

        Args:
            base_url: Base URL. Type is str. No default (must be specified).
            authentication_type: Authentication type. Type is AuthenticationType enum (API_KEY, SECURITY_TOKEN, INSTANCE_PRINCIPAL, or RESOURCE_PRINCIPAL). Default is SECURITY_TOKEN.
            authentication_configuration_file_spec: Authentication configuration file spec. Type is str. Default is "~/.oci/config". Applies only for API_KEY or SECURITY_TOKEN.
            authentication_profile_name: Authentication profile name. Type is str. Default is "DEFAULT". Applies only for API_KEY or SECURITY_TOKEN.
            back_end: Back-end. Type is BackEnd enum (GEN_AI_LLM or GEN_AI_AGENT). Default is GEN_AI_LLM.
            compartment_id: Compartment ID. Type is str. Default is None (must be specified). Applies only for GEN_AI_LLM.
            model_type: Model type. Type is str. Default is "GENERIC". Must be one of "GENERIC" or "COHERE". Applies only for GEN_AI_LLM.
            model_id: Model ID. Type is str. Default is None (must be specified or model_name must be specified). Applies only for GEN_AI_LLM.
            model_name: Model name. Type is name. Default is None (must be specified or model_id must be specified). Applies only for GEN_AI_LLM.
            maximum_number_of_tokens: Maximum number of tokens. Type is int. Default is None. Applies only for GEN_AI_LLM.
            temperature: Temperature. Type is float. Default is None. Applies only for GEN_AI_LLM.
            top_p: Top-P. Type is float. Default is None. Applies only for GEN_AI_LLM.
            top_k: Top-K. Type is int. Default is None. Applies only for GEN_AI_LLM.
            frequency_penalty: Frequency penalty. Type is float. Default is None. Applies only for GEN_AI_LLM.
            presence_penalty: Presence penalty. Type is float. Default is None. Applies only for GEN_AI_LLM.
            seed: Seed. Type is int. Default is None. Applies only for GEN_AI_LLM.
            agent_endpoint_id: Agent endpoint ID. Type is str. Default is None (must be specified). Applies only for GEN_AI_AGENT.
        """

        super().__init__()

        self._oracle_llm = OracleLLM(
            base_url=base_url,
            authentication_type=authentication_type,
            authentication_configuration_file_spec=authentication_configuration_file_spec,
            authentication_profile_name=authentication_profile_name,
            back_end=back_end,
            compartment_id=compartment_id,
            model_type=model_type,
            model_id=model_id,
            model_name=model_name,
            maximum_number_of_tokens=maximum_number_of_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            agent_endpoint_id=agent_endpoint_id,
        )

        #
        #  currently this is never cleaned up because it appears that the past tool calls may
        #  always be needed to construct the entire conversation history. if this is not actually
        #  the case, theoretically old keys that are no longer referenced should be removed.
        #
        self._call_id_to_tool_call_dictionary = {}

        logger.debug("Initialized LLM.")

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        tools=None,
        tool_choice=None,
        extra_kwargs=None,
    ) -> LLMStream:
        return LLMStream(
            oracle_llm_livekit_plugin=self,
            chat_ctx=chat_ctx,
            conn_options=conn_options,
            tools=tools,
        )


class LLMStream(llm.LLMStream):
    """
    The LLM stream class. This derives from livekit.agents.llm.LLMStream.
    """

    def __init__(
        self,
        *,
        oracle_llm_livekit_plugin: LLM,
        chat_ctx: llm.ChatContext,
        conn_options: None,
        tools: None,
    ) -> None:
        super().__init__(
            oracle_llm_livekit_plugin, chat_ctx=chat_ctx, tools=None, conn_options=conn_options
        )

        self._oracle_llm_livekit_plugin = oracle_llm_livekit_plugin

        self._tools = LLMStream.convert_tools(tools)

        logger.debug("Converted tools.")

    async def _run(self) -> None:
        oracle_llm_content_list = []

        for chat_message in self._chat_ctx._items:
            if chat_message.type == "message":
                role = Role(chat_message.role.upper())
                for message in chat_message.content:
                    oracle_llm_content = OracleLLMContent(message, CONTENT_TYPE_STRING, role)
                    oracle_llm_content_list.append(oracle_llm_content)

            elif chat_message.type == "function_call_output":
                call_id = chat_message.call_id

                tool_call = self._oracle_llm_livekit_plugin._call_id_to_tool_call_dictionary.get(
                    call_id
                )

                if tool_call is not None:
                    try:
                        output_json = json.loads(chat_message.output)
                        message = output_json["text"]
                    except Exception:
                        message = chat_message.output

                    oracle_llm_content = OracleLLMContent(
                        tool_call, CONTENT_TYPE_STRING, Role.ASSISTANT
                    )
                    oracle_llm_content_list.append(oracle_llm_content)

                    oracle_llm_content = OracleLLMContent(
                        "The function result of " + tool_call + " is: " + message,
                        CONTENT_TYPE_STRING,
                        Role.SYSTEM,
                    )
                    oracle_llm_content_list.append(oracle_llm_content)

        logger.debug(
            "Before running content thru LLM. Content list count: "
            + str(len(oracle_llm_content_list))
        )

        response_messages = self._oracle_llm_livekit_plugin._oracle_llm.run(
            oracle_llm_content_list=oracle_llm_content_list, tools=self._tools
        )

        logger.debug(
            "After running content thru LLM. Response message list count: "
            + str(len(response_messages))
            + "."
        )

        for response_message in response_messages:
            if response_message.startswith(TOOL_CALL_PREFIX):
                tool_call = response_message

                logger.debug("External tool call needs to be made: " + tool_call)

                function_name, function_parameters = (
                    LLMStream.get_name_and_arguments_from_tool_call(tool_call)
                )

                tool = None
                for temp_tool in self._tools:
                    if temp_tool.name == function_name and len(temp_tool.parameters) == len(
                        function_parameters
                    ):
                        tool = temp_tool

                if tool is None:
                    raise Exception(
                        "Unknown function name: "
                        + function_name
                        + " in "
                        + TOOL_CALL_DESCRIPTION
                        + " response message: "
                        + tool_call
                        + "."
                    )

                function_parameters_text = "{"
                for i in range(len(function_parameters)):
                    parameter = tool.parameters[i]
                    if i > 0:
                        function_parameters_text += ","
                    function_parameters_text += '"' + parameter.name + '":'
                    is_string_parameter = parameter.type in {"string", "str"}
                    if is_string_parameter:
                        function_parameters_text += '"'
                    function_parameters_text += str(function_parameters[i])
                    if is_string_parameter:
                        function_parameters_text += '"'
                function_parameters_text += "}"

                call_id = utils.shortuuid()

                self._oracle_llm_livekit_plugin._call_id_to_tool_call_dictionary[call_id] = (
                    tool_call
                )

                function_tool_call = llm.FunctionToolCall(
                    name=function_name, arguments=function_parameters_text, call_id=call_id
                )

                choice_delta = llm.ChoiceDelta(
                    role=Role.ASSISTANT.name.lower(), content=None, tool_calls=[function_tool_call]
                )

                chat_chunk = llm.ChatChunk(id=utils.shortuuid(), delta=choice_delta, usage=None)

                self._event_ch.send_nowait(chat_chunk)

                logger.debug("Added tool call to event channel: " + tool_call)

            else:
                logger.debug("LLM response message: " + response_message)

                chat_chunk = llm.ChatChunk(
                    id=utils.shortuuid(),
                    delta=llm.ChoiceDelta(
                        content=response_message, role=Role.ASSISTANT.name.lower()
                    ),
                )

                self._event_ch.send_nowait(chat_chunk)

                logger.debug("Added response message to event channel: " + response_message)

    @staticmethod
    def convert_tools(livekit_tools):
        tools = []

        if livekit_tools is not None:
            function_contexts = to_fnc_ctx(livekit_tools)

            for function_context in function_contexts:
                type = function_context["type"]
                if type == "function":
                    function = function_context["function"]

                    function_name = function["name"]
                    function_description = function["description"]
                    if function_description is None or len(function_description) == 0:
                        function_description = function_name

                    function_parameters = function["parameters"]

                    parameters = []
                    for property_key, property_value in function_parameters["properties"].items():
                        parameter_name = property_key
                        if "description" in property_value:
                            parameter_description = property_value["description"]
                        elif "title" in property_value:
                            parameter_description = property_value["title"]
                        else:
                            parameter_description = parameter_name
                        parameter_type = property_value["type"]

                        parameter = OracleValue(
                            parameter_name, parameter_description, parameter_type
                        )
                        parameters.append(parameter)

                    tool = OracleTool(function_name, function_description, parameters)
                    tools.append(tool)

        if len(tools) == 0:
            return None

        return tools

    @staticmethod
    def get_name_and_arguments_from_tool_call(tool_call):
        tool_call = tool_call[len(TOOL_CALL_PREFIX) :].strip()

        function_name, function_parameters = LLMStream.parse_function_call(
            tool_call, TOOL_CALL_DESCRIPTION
        )

        return function_name, function_parameters

    @staticmethod
    def parse_function_call(code_string, description):
        expression = ast.parse(code_string, mode="eval").body

        if not isinstance(expression, ast.Call):
            raise Exception("Invalid " + description + ": " + code_string + ".")

        function_name = expression.func.id if isinstance(expression.func, ast.Name) else None
        if not function_name:
            raise Exception("Invalid " + description + ": " + code_string + ".")

        function_parameters = [ast.literal_eval(parameter) for parameter in expression.args]

        return function_name, function_parameters
