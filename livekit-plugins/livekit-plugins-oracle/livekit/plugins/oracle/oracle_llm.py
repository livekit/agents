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
This module wraps Oracle's LLM cloud service. While it is used by the Oracle LiveKit LLM plug-in,
it it completely indpendent of LiveKit and could be used in other environments besides LiveKit.

Author: Keith Schnable (at Oracle Corporation)
Date: 2025-08-12
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import oci

from .log import logger
from .utils import AuthenticationType, get_config_and_signer


class BackEnd(Enum):
    """Back-ends as enumerator."""

    GEN_AI_LLM = "GEN_AI_LLM"
    GEN_AI_AGENT = "GEN_AI_AGENT"


CONTENT_TYPE_STRING = "string"


class Role(Enum):
    """Roles as enumerator."""

    USER = "USER"
    SYSTEM = "SYSTEM"
    ASSISTANT = "ASSISTANT"
    DEVELOPER = "DEVELOPER"


TOOL_CALL_PREFIX = "TOOL-CALL:"
TOOL_CALL_DESCRIPTION = "tool-call"


class OracleLLM:
    """
    The Oracle LLM class. This class wraps the Oracle LLM service.
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
        Create a new instance of the OracleLLM class to access Oracle's GenAI service. This has no LiveKit dependencies.

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

        self._parameters = Parameters()

        self._parameters.base_url = base_url
        self._parameters.authentication_type = authentication_type
        self._parameters.authentication_configuration_file_spec = (
            authentication_configuration_file_spec
        )
        self._parameters.authentication_profile_name = authentication_profile_name
        self._parameters.back_end = back_end

        self._parameters.compartment_id = compartment_id
        self._parameters.model_type = model_type
        self._parameters.model_id = model_id
        self._parameters.model_name = model_name
        self._parameters.maximum_number_of_tokens = maximum_number_of_tokens
        self._parameters.temperature = temperature
        self._parameters.top_p = top_p
        self._parameters.top_k = top_k
        self._parameters.frequency_penalty = frequency_penalty
        self._parameters.presence_penalty = presence_penalty
        self._parameters.seed = seed

        self._parameters.agent_endpoint_id = agent_endpoint_id

        self.validate_parameters()

        self._number_of_runs = 0

        self._output_tool_descriptions = True

        if self._parameters.back_end == BackEnd.GEN_AI_LLM:
            self.initialize_for_llm()
        else:  # if self._parameters.back_end == BackEnd.GEN_AI_AGENT:
            self.initialize_for_agent()

        logger.debug("Initialized OracleLLM.")

    def validate_parameters(self):
        if not isinstance(self._parameters.base_url, str):
            raise TypeError("The base_url parameter must be a string.")
        self._parameters.base_url = self._parameters.base_url.strip()
        parsed = urlparse(self._parameters.base_url)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("The base_url parameter must be a valid URL.")

        if not isinstance(self._parameters.authentication_type, AuthenticationType):
            raise TypeError(
                "The authentication_type parameter must be one of the AuthenticationType enum members."
            )

        if self._parameters.authentication_type in {
            AuthenticationType.API_KEY,
            AuthenticationType.SECURITY_TOKEN,
        }:
            if not isinstance(self._parameters.authentication_configuration_file_spec, str):
                raise TypeError(
                    "The authentication_configuration_file_spec parameter must be a string."
                )
            self._parameters.authentication_configuration_file_spec = (
                self._parameters.authentication_configuration_file_spec.strip()
            )
            if len(self._parameters.authentication_configuration_file_spec) == 0:
                raise ValueError(
                    "The authentication_configuration_file_spec parameter must not be an empty string."
                )

            if not isinstance(self._parameters.authentication_profile_name, str):
                raise TypeError("The authentication_profile_name parameter must be a string.")
            self._parameters.authentication_profile_name = (
                self._parameters.authentication_profile_name.strip()
            )
            if len(self._parameters.authentication_profile_name) == 0:
                raise ValueError(
                    "The authentication_profile_name parameter must not be an empty string."
                )

        if not isinstance(self._parameters.back_end, BackEnd):
            raise TypeError("The back_end parameter must be one of the BackEnd enum members.")

        if self._parameters.back_end == BackEnd.GEN_AI_LLM:
            if not isinstance(self._parameters.compartment_id, str):
                raise TypeError("The compartment_id parameter must be a string.")
            self._parameters.compartment_id = self._parameters.compartment_id.strip()
            if len(self._parameters.compartment_id) == 0:
                raise ValueError("The compartment_id parameter must not be an empty string.")

            if not isinstance(self._parameters.model_type, str):
                raise TypeError("The model_type parameter must be a string.")
            self._parameters.model_type = self._parameters.model_type.strip().upper()
            if self._parameters.model_type not in {"GENERIC", "COHERE"}:
                raise ValueError("The model_type parameter must be 'GENERIC' or 'COHERE'.")

            if self._parameters.model_id is not None:
                if not isinstance(self._parameters.model_id, str):
                    raise TypeError("The model_id parameter must be a string.")
                self._parameters.model_id = self._parameters.model_id.strip()
                if len(self._parameters.model_id) == 0:
                    raise ValueError("The model_id parameter must not be an empty string.")

            if self._parameters.model_name is not None:
                if not isinstance(self._parameters.model_name, str):
                    raise TypeError("The model_name parameter must be a string.")
                self._parameters.model_name = self._parameters.model_name.strip()
                if len(self._parameters.model_name) == 0:
                    raise ValueError("The model_name parameter must not be an empty string.")

            if self._parameters.model_id is None:
                if self._parameters.model_name is None:
                    raise TypeError(
                        "Either the model_id or the model_name parameter must not be None."
                    )
            elif self._parameters.model_name is not None:
                raise TypeError("Either the model_id or the model_name parameter must be None.")

            if self._parameters.maximum_number_of_tokens is not None:
                if not isinstance(self._parameters.maximum_number_of_tokens, int):
                    raise TypeError("The maximum_number_of_tokens parameter must be an integer.")
                if self._parameters.maximum_number_of_tokens <= 0:
                    raise ValueError(
                        "The maximum_number_of_tokens parameter must be greater than 0."
                    )

            if self._parameters.temperature is not None:
                if not isinstance(self._parameters.temperature, float):
                    raise TypeError("The temperature parameter must be a float.")
                if self._parameters.temperature < 0:
                    raise ValueError(
                        "The maximum_number_of_tokens parameter must be greater than or equal to 0."
                    )

            if self._parameters.top_p is not None:
                if not isinstance(self._parameters.top_p, float):
                    raise TypeError("The top_p parameter must be a float.")
                if self._parameters.top_p < 0 or self._parameters.top_p > 1:
                    raise ValueError("The top_p parameter must be between 0 and 1.")

            if self._parameters.top_k is not None:
                if not isinstance(self._parameters.top_k, int):
                    raise TypeError("The top_k parameter must be an integer.")
                if self._parameters.top_k <= 0:
                    raise ValueError("The top_k parameter must be greater than 0.")

            if self._parameters.frequency_penalty is not None:
                if not isinstance(self._parameters.frequency_penalty, float):
                    raise TypeError("The frequency_penalty parameter must be a float.")
                if self._parameters.frequency_penalty < 0:
                    raise ValueError(
                        "The frequency_penalty parameter must be greater than or equal to 0."
                    )

            if self._parameters.presence_penalty is not None:
                if not isinstance(self._parameters.presence_penalty, float):
                    raise TypeError("The presence_penalty parameter must be a float.")
                if self._parameters.presence_penalty < 0:
                    raise ValueError(
                        "The presence_penalty parameter must be greater than or equal to 0."
                    )

            if self._parameters.seed is not None:  # noqa: SIM102
                if not isinstance(self._parameters.seed, int):
                    raise TypeError("The seed parameter must be an integer.")

        elif self._parameters.back_end == BackEnd.GEN_AI_AGENT:
            if not isinstance(self._parameters.agent_endpoint_id, str):
                raise TypeError("The agent_endpoint_id parameter must be a string.")
            self._parameters.agent_endpoint_id = self._parameters.agent_endpoint_id.strip()
            if len(self._parameters.agent_endpoint_id) == 0:
                raise ValueError("The agent_endpoint_id parameter must not be an empty string.")

    def initialize_for_llm(self):
        configAndSigner = get_config_and_signer(
            authentication_type=self._parameters.authentication_type,
            authentication_configuration_file_spec=self._parameters.authentication_configuration_file_spec,
            authentication_profile_name=self._parameters.authentication_profile_name,
        )
        config = configAndSigner["config"]
        signer = configAndSigner["signer"]

        if signer is None:
            self._generative_ai_inference_client = (
                oci.generative_ai_inference.GenerativeAiInferenceClient(
                    config=config,
                    service_endpoint=self._parameters.base_url,
                    retry_strategy=oci.retry.NoneRetryStrategy(),
                )
            )
        else:
            self._generative_ai_inference_client = (
                oci.generative_ai_inference.GenerativeAiInferenceClient(
                    config=config,
                    service_endpoint=self._parameters.base_url,
                    retry_strategy=oci.retry.NoneRetryStrategy(),
                    signer=signer,
                )
            )

        logger.debug("Initialized for GenAI LLM.")

    def initialize_for_agent(self):
        configAndSigner = get_config_and_signer(
            authentication_type=self._parameters.authentication_type,
            authentication_configuration_file_spec=self._parameters.authentication_configuration_file_spec,
            authentication_profile_name=self._parameters.authentication_profile_name,
        )
        config = configAndSigner["config"]
        signer = configAndSigner["signer"]

        if signer is None:
            self._generative_ai_agent_runtime_client = (
                oci.generative_ai_agent_runtime.GenerativeAiAgentRuntimeClient(
                    config=config,
                    service_endpoint=self._parameters.base_url,
                    retry_strategy=oci.retry.NoneRetryStrategy(),
                )
            )
        else:
            self._generative_ai_agent_runtime_client = (
                oci.generative_ai_agent_runtime.GenerativeAiAgentRuntimeClient(
                    config=config,
                    service_endpoint=self._parameters.base_url,
                    retry_strategy=oci.retry.NoneRetryStrategy(),
                    signer=signer,
                )
            )

        id = str(uuid.uuid4())

        session_details = oci.generative_ai_agent_runtime.models.CreateSessionDetails(
            display_name="display_name_for_" + id, description="description_for_" + id
        )

        response = self._generative_ai_agent_runtime_client.create_session(
            agent_endpoint_id=self._parameters.agent_endpoint_id,
            create_session_details=session_details,
        )
        self._session_id = response.data.id

        logger.debug("Initialized for GenAI Agent.")

    def run(
        self,
        *,
        oracle_llm_content_list: list[OracleLLMContent] = None,
        tools: list[OracleTool] = None,
    ) -> list[str]:
        if self._parameters.back_end == BackEnd.GEN_AI_LLM:
            response_messages = self.run_for_llm(
                oracle_llm_content_list=oracle_llm_content_list, tools=tools
            )
        else:  # if self._parameters.back_end == BackEnd.GEN_AI_AGENT:
            response_messages = self.run_for_agent(
                oracle_llm_content_list=oracle_llm_content_list, tools=tools
            )

        self._number_of_runs += 1

        return response_messages

    def run_for_llm(
        self,
        *,
        oracle_llm_content_list: list[OracleLLMContent] = None,
        tools: list[OracleTool] = None,
    ) -> list[str]:
        if oracle_llm_content_list is None:
            oracle_llm_content_list = []

        temp_message_list = []
        temp_messages = ""

        tool_descriptions = self.get_tool_descriptions(tools)
        if tool_descriptions is not None:
            text_content = oci.generative_ai_inference.models.TextContent()
            text_content.text = tool_descriptions

            message = oci.generative_ai_inference.models.Message()
            message.role = Role.SYSTEM.name
            message.content = [text_content]

            temp_message_list.append(message)

            if len(temp_messages) > 0:
                temp_messages += "\n"

            temp_messages += tool_descriptions

        for oracle_llm_content in oracle_llm_content_list:
            if oracle_llm_content.content_type == CONTENT_TYPE_STRING:
                text_content = oci.generative_ai_inference.models.TextContent()
                text_content.text = oracle_llm_content.content_data

                message = oci.generative_ai_inference.models.Message()
                message.role = oracle_llm_content.role.name
                message.content = [text_content]

                temp_message_list.append(message)

                if len(temp_messages) > 0:
                    temp_messages += "\n"

                temp_messages += oracle_llm_content.content_data

        if self._parameters.model_type == "GENERIC":
            chat_request = oci.generative_ai_inference.models.GenericChatRequest()
            chat_request.messages = temp_message_list

        elif self._parameters.model_type == "COHERE":
            chat_request = oci.generative_ai_inference.models.CohereChatRequest()
            chat_request.message = temp_messages

        if self._parameters.maximum_number_of_tokens is not None:
            chat_request.max_tokens = self._parameters.maximum_number_of_tokens
        if self._parameters.temperature is not None:
            chat_request.temperature = self._parameters.temperature
        if self._parameters.frequency_penalty is not None:
            chat_request.frequency_penalty = self._parameters.frequency_penalty
        if self._parameters.presence_penalty is not None:
            chat_request.presence_penalty = self._parameters.presence_penalty
        if self._parameters.top_p is not None:
            chat_request.top_p = self._parameters.top_p
        if self._parameters.top_k is not None:
            chat_request.top_k = self._parameters.top_k
        if self._parameters.seed is not None:
            chat_request.seed = self._parameters.seed

        serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=self._parameters.model_name
            if self._parameters.model_id is None
            else self._parameters.model_id
        )

        chat_details = oci.generative_ai_inference.models.ChatDetails()
        chat_details.serving_mode = serving_mode
        chat_details.chat_request = chat_request
        chat_details.compartment_id = self._parameters.compartment_id

        logger.debug("Before calling GenAI LLM.")

        chat_response = self._generative_ai_inference_client.chat(chat_details)

        logger.debug("After calling GenAI LLM.")

        if self._parameters.model_type == "GENERIC":
            response_messages = []
            for temp_content in chat_response.data.chat_response.choices[0].message.content:
                response_messages.append(temp_content.text)
        elif self._parameters.model_type == "COHERE":
            response_messages = [chat_response.data.chat_response.text]

        new_response_messages = []

        for response_message in response_messages:
            logger.debug("Raw response message: " + response_message)
            tool_call_index = response_message.find(TOOL_CALL_PREFIX)
            if tool_call_index == -1:
                new_response_messages.append(response_message)
                logger.debug("Response message: " + response_message)
            else:
                tool_call = response_message[tool_call_index:]
                if tool_call_index != 0:
                    response_message = response_message[:tool_call_index]
                    new_response_messages.append(response_message)
                    logger.debug("Response message: " + response_message)
                new_response_messages.append(tool_call)
                logger.debug("Tool call: " + tool_call)

        response_messages = new_response_messages

        return response_messages

    def run_for_agent(
        self,
        *,
        oracle_llm_content_list: list[OracleLLMContent] = None,
        tools: list[OracleTool] = None,
    ) -> list[str]:
        if oracle_llm_content_list is None:
            oracle_llm_content_list = []

        user_message = ""

        if self._number_of_runs == 0:
            tool_descriptions = self.get_tool_descriptions(tools)
            if tool_descriptions is not None:
                if len(user_message) > 0:
                    user_message += "\n"
                user_message += tool_descriptions

        for oracle_llm_content in reversed(oracle_llm_content_list):
            if oracle_llm_content.content_type == CONTENT_TYPE_STRING:
                if len(user_message) > 0:
                    user_message += "\n"
                user_message += oracle_llm_content.content_data
                break

        logger.debug(user_message)

        chat_details = oci.generative_ai_agent_runtime.models.ChatDetails(
            session_id=self._session_id, user_message=user_message, should_stream=False
        )

        logger.debug("Before calling GenAI agent.")

        response = self._generative_ai_agent_runtime_client.chat(
            agent_endpoint_id=self._parameters.agent_endpoint_id, chat_details=chat_details
        )

        logger.debug("After calling GenAI agent.")

        response_message = response.data.message.content.text

        logger.debug(response_message)

        response_messages = [response_message]

        if (
            TOOL_CALL_PREFIX in response_message
            and response_message.find(TOOL_CALL_PREFIX, 1) != -1
        ):
            raise Exception(
                "Unexpectedly received a response message with an embedded "
                + TOOL_CALL_DESCRIPTION
                + "."
            )

        return response_messages

    def get_tool_descriptions(self, tools):
        if tools is None or len(tools) == 0:
            return None

        tool_descriptions = "You are an assistant with access to the following functions:\n\n"

        for i in range(len(tools)):
            tool = tools[i]

            tool_descriptions += str(i + 1) + ". The function prototype is: " + tool.name + "("

            for j in range(len(tool.parameters)):
                parameter = tool.parameters[j]
                if j > 0:
                    tool_descriptions += ","
                tool_descriptions += parameter.name

            tool_descriptions += ") and the function description is: " + tool.description + "\n"

        tool_descriptions += (
            '\nAlways indicate when you want to call a function by writing: "'
            + TOOL_CALL_PREFIX
            + ' function_name(parameters)"\n'
        )
        tool_descriptions += "Do not combine function calls and text responses in the same output: either only function calls or only text responses.\n"
        tool_descriptions += (
            "For any string parameters, be sure to enclose each of them in double quotes."
        )

        if self._output_tool_descriptions:
            self._output_tool_descriptions = False
            logger.debug(tool_descriptions)

        return tool_descriptions


class Parameters:
    """
    The parameters class. This class contains all parameter information for the Oracle LLM class.
    """

    base_url: str

    back_end: str

    compartment_id: str
    authentication_type: AuthenticationType
    authentication_configuration_file_spec: str
    authentication_profile_name: str
    model_type: str
    model_id: str
    model_name: str
    maximum_number_of_tokens: int
    temperature: float
    top_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    seed: int

    agent_endpoint_id: str


@dataclass
class OracleLLMContent:
    """
    The Oracle LLM content class. This class contains all information related to one LLM content item.
    """

    content_data: Any
    content_type: str
    role: Role


@dataclass
class OracleValue:
    """
    The Oracle value class. This class contains all information related to one value.
    """

    name: str
    description: str
    type: str


@dataclass
class OracleTool:
    """
    The Oracle tool class. This class contains all information related to one tool.
    """

    name: str
    description: str
    parameters: list[OracleValue]
