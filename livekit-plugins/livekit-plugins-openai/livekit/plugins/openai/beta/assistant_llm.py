# Copyright 2023 LiveKit, Inc.
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

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, MutableSet, Union

import httpx
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm import ToolChoice
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from openai import AsyncAssistantEventHandler, AsyncClient
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.run_create_params import AdditionalMessage
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from openai.types.beta.threads.runs import (
    CodeInterpreterToolCall,
    FileSearchToolCall,
    FunctionToolCall,
    ToolCall,
)
from openai.types.file_object import FileObject

from .._oai_api import build_oai_function_description
from ..log import logger
from ..models import ChatModels

DEFAULT_MODEL = "gpt-4o"
OPENAI_MESSAGE_ID_KEY = "__openai_message_id__"
LIVEKIT_MESSAGE_ID_KEY = "__livekit_message_id__"
OPENAI_MESSAGES_ADDED_KEY = "__openai_messages_added__"
OPENAI_FILE_ID_KEY = "__openai_file_id__"


@dataclass
class LLMOptions:
    model: str | ChatModels


@dataclass
class AssistantOptions:
    """Options for creating (on-the-fly) or loading an assistant. Only one of create_options or load_options should be set."""

    create_options: AssistantCreateOptions | None = None
    load_options: AssistantLoadOptions | None = None


@dataclass
class AssistantCreateOptions:
    name: str
    instructions: str
    model: ChatModels
    temperature: float | None = None
    # TODO: when we implement code_interpreter and file_search tools
    # tool_resources: ToolResources | None = None
    # tools: list[AssistantTools] = field(default_factory=list)


@dataclass
class AssistantLoadOptions:
    assistant_id: str
    thread_id: str | None


@dataclass
class OnFileUploadedInfo:
    type: Literal["image"]
    original_file: llm.ChatImage
    openai_file_object: FileObject


OnFileUploaded = Callable[[OnFileUploadedInfo], None]


class AssistantLLM(llm.LLM):
    def __init__(
        self,
        *,
        assistant_opts: AssistantOptions,
        client: AsyncClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        on_file_uploaded: OnFileUploaded | None = None,
    ) -> None:
        super().__init__()

        test_ctx = llm.ChatContext()
        if not hasattr(test_ctx, "_metadata"):
            raise Exception(
                "This beta feature of 'livekit-plugins-openai' requires a newer version of 'livekit-agents'"
            )
        self._client = client or AsyncClient(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=30, connect=10, read=5, pool=5),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                    keepalive_expiry=120,
                ),
            ),
        )
        self._assistant_opts = assistant_opts
        self._running_fncs: MutableSet[asyncio.Task[Any]] = set()
        self._on_file_uploaded = on_file_uploaded
        self._tool_call_run_id_lookup = dict[str, str]()
        self._submitted_tool_calls = set[str]()

        self._sync_openai_task: asyncio.Task[AssistantLoadOptions] | None = None
        try:
            self._sync_openai_task = asyncio.create_task(self._sync_openai())
        except Exception:
            logger.error(
                "failed to create sync openai task. This can happen when instantiating without a running asyncio event loop (such has when running tests)"
            )
        self._done_futures = list[asyncio.Future[None]]()

    async def _sync_openai(self) -> AssistantLoadOptions:
        if self._assistant_opts.create_options:
            kwargs: Dict[str, Any] = {
                "model": self._assistant_opts.create_options.model,
                "name": self._assistant_opts.create_options.name,
                "instructions": self._assistant_opts.create_options.instructions,
                # "tools": [
                #     {"type": t} for t in self._assistant_opts.create_options.tools
                # ],
                # "tool_resources": self._assistant_opts.create_options.tool_resources,
            }
            # TODO when we implement code_interpreter and file_search tools
            # if self._assistant_opts.create_options.tool_resources:
            #     kwargs["tool_resources"] = (
            #         self._assistant_opts.create_options.tool_resources
            #     )
            if self._assistant_opts.create_options.temperature:
                kwargs["temperature"] = self._assistant_opts.create_options.temperature
            assistant = await self._client.beta.assistants.create(**kwargs)

            thread = await self._client.beta.threads.create()
            return AssistantLoadOptions(assistant_id=assistant.id, thread_id=thread.id)
        elif self._assistant_opts.load_options:
            if not self._assistant_opts.load_options.thread_id:
                thread = await self._client.beta.threads.create()
                self._assistant_opts.load_options.thread_id = thread.id
            return self._assistant_opts.load_options

        raise Exception("One of create_options or load_options must be set")

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]]
        | None = None,
    ):
        if n is not None:
            logger.warning("OpenAI Assistants does not support the 'n' parameter")

        if parallel_tool_calls is not None:
            logger.warning(
                "OpenAI Assistants does not support the 'parallel_tool_calls' parameter"
            )

        if not self._sync_openai_task:
            self._sync_openai_task = asyncio.create_task(self._sync_openai())

        return AssistantLLMStream(
            temperature=temperature,
            assistant_llm=self,
            sync_openai_task=self._sync_openai_task,
            client=self._client,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            on_file_uploaded=self._on_file_uploaded,
            conn_options=conn_options,
        )

    async def _register_tool_call(self, tool_call_id: str, run_id: str) -> None:
        self._tool_call_run_id_lookup[tool_call_id] = run_id

    async def _submit_tool_call_result(self, tool_call_id: str, result: str) -> None:
        if tool_call_id in self._submitted_tool_calls:
            return
        logger.debug(f"submitting tool call {tool_call_id} result")
        run_id = self._tool_call_run_id_lookup.get(tool_call_id)
        if not run_id:
            logger.error(f"tool call {tool_call_id} not found")
            return

        if not self._sync_openai_task:
            logger.error("sync_openai_task not set")
            return

        thread_id = (await self._sync_openai_task).thread_id
        if not thread_id:
            logger.error("thread_id not set")
            return
        tool_output = ToolOutput(output=result, tool_call_id=tool_call_id)
        await self._client.beta.threads.runs.submit_tool_outputs_and_poll(
            tool_outputs=[tool_output], run_id=run_id, thread_id=thread_id
        )
        self._submitted_tool_calls.add(tool_call_id)
        logger.debug(f"submitted tool call {tool_call_id} result")


class AssistantLLMStream(llm.LLMStream):
    class EventHandler(AsyncAssistantEventHandler):
        def __init__(
            self,
            llm: AssistantLLM,
            llm_stream: AssistantLLMStream,
            event_ch: utils.aio.Chan[llm.ChatChunk],
            chat_ctx: llm.ChatContext,
            fnc_ctx: llm.FunctionContext | None = None,
        ):
            super().__init__()
            self._llm = llm
            self._llm_stream = llm_stream
            self._chat_ctx = chat_ctx
            self._event_ch = event_ch
            self._fnc_ctx = fnc_ctx

        async def on_text_delta(self, delta: TextDelta, snapshot: Text):
            assert self.current_run is not None

            self._event_ch.send_nowait(
                llm.ChatChunk(
                    request_id=self.current_run.id,
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(role="assistant", content=delta.value)
                        )
                    ],
                )
            )

        async def on_tool_call_created(self, tool_call: ToolCall):
            if not self.current_run:
                logger.error("tool call created without run")
                return
            await self._llm._register_tool_call(tool_call.id, self.current_run.id)

        async def on_tool_call_done(
            self,
            tool_call: CodeInterpreterToolCall | FileSearchToolCall | FunctionToolCall,
        ) -> None:
            assert self.current_run is not None

            if tool_call.type == "code_interpreter":
                logger.warning("code interpreter tool call not yet implemented")
            elif tool_call.type == "file_search":
                logger.warning("file_search tool call not yet implemented")
            elif tool_call.type == "function":
                if not self._fnc_ctx:
                    logger.error("function tool called without function context")
                    return

                fnc = llm.FunctionCallInfo(
                    function_info=self._fnc_ctx.ai_functions[tool_call.function.name],
                    arguments=json.loads(tool_call.function.arguments),
                    tool_call_id=tool_call.id,
                    raw_arguments=tool_call.function.arguments,
                )

                self._llm_stream._function_calls_info.append(fnc)
                chunk = llm.ChatChunk(
                    request_id=self.current_run.id,
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(role="assistant", tool_calls=[fnc]),
                            index=0,
                        )
                    ],
                )
                self._event_ch.send_nowait(chunk)

    def __init__(
        self,
        *,
        assistant_llm: AssistantLLM,
        client: AsyncClient,
        sync_openai_task: asyncio.Task[AssistantLoadOptions],
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        temperature: float | None,
        on_file_uploaded: OnFileUploaded | None,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            assistant_llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, conn_options=conn_options
        )
        self._client = client
        self._temperature = temperature
        self._on_file_uploaded = on_file_uploaded

        # current function call that we're waiting for full completion (args are streamed)
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self._create_stream_task = asyncio.create_task(self._main_task())
        self._sync_openai_task = sync_openai_task

        # Running stream is used to ensure that we only have one stream running at a time
        self._done_future: asyncio.Future[None] = asyncio.Future()

    async def _run(self) -> None:
        assert isinstance(self._llm, AssistantLLM)

        # This function's complexity is due to the fact that we need to sync chat_ctx messages with OpenAI.
        # OpenAI also does not allow us to modify messages while a stream is running. So we need to make sure streams run
        # sequentially. The strategy is as follows:
        #
        # 1. ensure that we have a thread_id and assistant_id from OpenAI. This comes from the _sync_openai_task
        # 2. make sure all previous streams are done before starting a new one
        # 3. delete messages that are no longer in the chat_ctx but are still in OpenAI by using the OpenAI message id
        # 4. add new messages to OpenAI that are in the chat_ctx but not in OpenAI. We don't know the OpenAI message id yet
        #    so we create a random uuid (we call it the LiveKit message id) and set that in the metdata.
        # 5. start the stream and wait for it to finish
        # 6. get the OpenAI message ids for the messages we added to OpenAI by using the metadata
        # 7. Resolve the OpenAI message id with all messages that have a LiveKit message id.
        try:
            load_options = await self._sync_openai_task

            # The assistants api does not let us modify messages while a stream is running.
            # So we have to make sure previous streams are done before starting a new one.
            await asyncio.gather(*self._llm._done_futures)
            self._llm._done_futures.clear()
            self._llm._done_futures.append(self._done_future)

            # OpenAI required submitting tool call outputs manually. We iterate
            # tool outputs in the chat_ctx (from previous runs) and submit them
            # before continuing.
            for msg in self._chat_ctx.messages:
                if msg.role == "tool":
                    if not msg.tool_call_id:
                        logger.error("tool message without tool_call_id")
                        continue
                    if not isinstance(msg.content, str):
                        logger.error("tool message content is not str")
                        continue
                    await self._llm._submit_tool_call_result(
                        msg.tool_call_id, msg.content
                    )

            # At the chat_ctx level, create a map of thread_id to message_ids
            # This is used to keep track of which messages have been added to the thread
            # and which we may need to delete from OpenAI
            if OPENAI_MESSAGES_ADDED_KEY not in self._chat_ctx._metadata:
                self._chat_ctx._metadata[OPENAI_MESSAGES_ADDED_KEY] = dict()

            if (
                load_options.thread_id
                not in self._chat_ctx._metadata[OPENAI_MESSAGES_ADDED_KEY]
            ):
                self._chat_ctx._metadata[OPENAI_MESSAGES_ADDED_KEY][
                    load_options.thread_id
                ] = set()

            # Keep this handy to make the code more readable later on
            openai_addded_messages_set: set[str] = self._chat_ctx._metadata[
                OPENAI_MESSAGES_ADDED_KEY
            ][load_options.thread_id]

            # Keep track of messages that are no longer in the chat_ctx but are still in OpenAI
            # Note: Unfortuneately, this will add latency unfortunately. Usually it's just one message so we loop it but
            # it will create an extra round trip to OpenAI before being able to run inference.
            # TODO: parallelize it?
            for msg in self._chat_ctx.messages:
                msg_id = msg._metadata.get(OPENAI_MESSAGE_ID_KEY, {}).get(
                    load_options.thread_id
                )
                assert load_options.thread_id
                if msg_id and msg_id not in openai_addded_messages_set:
                    await self._client.beta.threads.messages.delete(
                        thread_id=load_options.thread_id,
                        message_id=msg_id,
                    )
                    logger.debug(
                        f"Deleted message '{msg_id}' in thread '{load_options.thread_id}'"
                    )
                    openai_addded_messages_set.remove(msg_id)

            # Upload any images in the chat_ctx that have not been uploaded to OpenAI
            for msg in self._chat_ctx.messages:
                if msg.role != "user":
                    continue

                if not isinstance(msg.content, list):
                    continue

                for cnt in msg.content:
                    if (
                        not isinstance(cnt, llm.ChatImage)
                        or OPENAI_FILE_ID_KEY in cnt._cache
                    ):
                        continue

                    if isinstance(cnt.image, str):
                        continue

                    file_obj = await self._upload_frame(
                        cnt.image, cnt.inference_width, cnt.inference_height
                    )
                    cnt._cache[OPENAI_FILE_ID_KEY] = file_obj.id
                    if self._on_file_uploaded:
                        self._on_file_uploaded(
                            OnFileUploadedInfo(
                                type="image",
                                original_file=cnt,
                                openai_file_object=file_obj,
                            )
                        )

            # Keep track of the new messages in the chat_ctx that we need to add to OpenAI
            additional_messages: list[AdditionalMessage] = []
            for msg in self._chat_ctx.messages:
                if msg.role != "user":
                    continue

                msg_id = str(uuid.uuid4())
                if OPENAI_MESSAGE_ID_KEY not in msg._metadata:
                    msg._metadata[OPENAI_MESSAGE_ID_KEY] = dict[str, str]()

                if LIVEKIT_MESSAGE_ID_KEY not in msg._metadata:
                    msg._metadata[LIVEKIT_MESSAGE_ID_KEY] = dict[str, str]()

                oai_msg_id_dict = msg._metadata[OPENAI_MESSAGE_ID_KEY]
                lk_msg_id_dict = msg._metadata[LIVEKIT_MESSAGE_ID_KEY]

                if load_options.thread_id not in oai_msg_id_dict:
                    converted_msg = build_oai_message(msg)
                    converted_msg["private_message_id"] = msg_id
                    additional_messages.append(
                        AdditionalMessage(
                            role="user",
                            content=converted_msg["content"],
                            metadata={LIVEKIT_MESSAGE_ID_KEY: msg_id},
                        )
                    )
                    lk_msg_id_dict[load_options.thread_id] = msg_id

            eh = AssistantLLMStream.EventHandler(
                llm=self._llm,
                event_ch=self._event_ch,
                chat_ctx=self._chat_ctx,
                fnc_ctx=self._fnc_ctx,
                llm_stream=self,
            )
            assert load_options.thread_id
            kwargs: dict[str, Any] = {
                "additional_messages": additional_messages,
                "thread_id": load_options.thread_id,
                "assistant_id": load_options.assistant_id,
                "event_handler": eh,
                "temperature": self._temperature,
            }
            if self._fnc_ctx:
                kwargs["tools"] = [
                    build_oai_function_description(f)
                    for f in self._fnc_ctx.ai_functions.values()
                ]

            async with self._client.beta.threads.runs.stream(**kwargs) as stream:
                await stream.until_done()

            # Populate the openai_message_id for the messages we added to OpenAI. Note, we do this after
            # sending None to close the iterator so that it is done in parellel with any users of
            # the stream. However, the next stream will not start until this is done.
            lk_to_oai_lookup = dict[str, str]()
            messages = await self._client.beta.threads.messages.list(
                thread_id=load_options.thread_id,
                limit=10,  # We could be smarter and make a more exact query, but this is probably fine
            )
            for oai_msg in messages.data:
                if oai_msg.metadata.get(LIVEKIT_MESSAGE_ID_KEY):  # type: ignore
                    lk_to_oai_lookup[oai_msg.metadata[LIVEKIT_MESSAGE_ID_KEY]] = (  # type: ignore
                        oai_msg.id
                    )

            for msg in self._chat_ctx.messages:
                if msg.role != "user":
                    continue
                oai_msg_id_dict = msg._metadata.get(OPENAI_MESSAGE_ID_KEY)
                lk_msg_id_dict = msg._metadata.get(LIVEKIT_MESSAGE_ID_KEY)
                if oai_msg_id_dict is None or lk_msg_id_dict is None:
                    continue

                lk_msg_id = lk_msg_id_dict.get(load_options.thread_id)
                if lk_msg_id and lk_msg_id in lk_to_oai_lookup:
                    oai_msg_id = lk_to_oai_lookup[lk_msg_id]
                    oai_msg_id_dict[load_options.thread_id] = oai_msg_id
                    openai_addded_messages_set.add(oai_msg_id)
                    # We don't need the LiveKit message id anymore
                    lk_msg_id_dict.pop(load_options.thread_id)

        finally:
            self._done_future.set_result(None)

    async def _upload_frame(
        self,
        frame: rtc.VideoFrame,
        inference_width: int | None,
        inference_height: int | None,
    ):
        # inside our internal implementation, we allow to put extra metadata to
        # each ChatImage (avoid to reencode each time we do a chatcompletion request)
        opts = utils.images.EncodeOptions()
        if inference_width and inference_height:
            opts.resize_options = utils.images.ResizeOptions(
                width=inference_width,
                height=inference_height,
                strategy="scale_aspect_fit",
            )

        encoded_data = utils.images.encode(frame, opts)
        fileObj = await self._client.files.create(
            file=("image.jpg", encoded_data),
            purpose="vision",
        )

        return fileObj


def build_oai_message(msg: llm.ChatMessage):
    oai_msg: dict[str, Any] = {"role": msg.role}

    if msg.name:
        oai_msg["name"] = msg.name

    # add content if provided
    if isinstance(msg.content, str):
        oai_msg["content"] = msg.content
    elif isinstance(msg.content, list):
        oai_content: list[dict[str, Any]] = []
        for cnt in msg.content:
            if isinstance(cnt, str):
                oai_content.append({"type": "text", "text": cnt})
            elif isinstance(cnt, llm.ChatImage):
                if cnt._cache[OPENAI_FILE_ID_KEY]:
                    oai_content.append(
                        {
                            "type": "image_file",
                            "image_file": {"file_id": cnt._cache[OPENAI_FILE_ID_KEY]},
                        }
                    )

        oai_msg["content"] = oai_content

    # make sure to provide when function has been called inside the context
    # (+ raw_arguments)
    if msg.tool_calls is not None:
        tool_calls: list[dict[str, Any]] = []
        oai_msg["tool_calls"] = tool_calls
        for fnc in msg.tool_calls:
            tool_calls.append(
                {
                    "id": fnc.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": fnc.function_info.name,
                        "arguments": fnc.raw_arguments,
                    },
                }
            )

    # tool_call_id is set when the message is a response/result to a function call
    # (content is a string in this case)
    if msg.tool_call_id:
        oai_msg["tool_call_id"] = msg.tool_call_id

    return oai_msg
