from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Union, Literal
import json

MEDIA_TYPE = Literal["text/plain", "audio/lpcm", "application/json"]
TYPE = Literal["TEXT", "AUDIO", "TOOL"]
VOICE_ID = Literal["matthew", "tiffany", "amy"]
ROLE = Literal["USER", "ASSISTANT", "TOOL", "SYSTEM"]
GENERATION_STAGE = Literal["SPECULATIVE", "FINAL"]
STOP_REASON = Literal["PARTIAL_TURN", "END_TURN", "INTERRUPTED"]
SAMPLE_RATE_HERTZ = Literal[8_000, 16_000, 24_000]
AUDIO_ENCODING = Literal["base64"]  # all audio data must be base64 encoded
SAMPLE_SIZE_BITS = Literal[16]  # only supports 16-bit audio
CHANNEL_COUNT = Literal[1]  # only supports monochannel audio


class BaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class InferenceConfiguration(BaseModel):
    maxTokens: int = Field(default=1024, ge=1, le=10_000, frozen=True)
    topP: float = Field(default=0.9, ge=0.0, le=1.0, frozen=True)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, frozen=True)


class AudioInputConfiguration(BaseModel):
    mediaType: MEDIA_TYPE = "audio/lpcm"
    sampleRateHertz: SAMPLE_RATE_HERTZ = Field(default=16000)
    sampleSizeBits: SAMPLE_SIZE_BITS = 16
    channelCount: CHANNEL_COUNT = 1
    audioType: str = "SPEECH"
    encoding: AUDIO_ENCODING = "base64"


class AudioOutputConfiguration(BaseModel):
    mediaType: MEDIA_TYPE = "audio/lpcm"
    sampleRateHertz: SAMPLE_RATE_HERTZ = Field(default=24_000)
    sampleSizeBits: SAMPLE_SIZE_BITS = 16
    channelCount: CHANNEL_COUNT = 1
    voiceId: VOICE_ID = Field(...)
    encoding: AUDIO_ENCODING = "base64"
    audioType: str = "SPEECH"


class TextInputConfiguration(BaseModel):
    mediaType: MEDIA_TYPE = "text/plain"


class TextOutputConfiguration(BaseModel):
    mediaType: MEDIA_TYPE = "text/plain"


class ToolUseOutputConfiguration(BaseModel):
    mediaType: MEDIA_TYPE = "application/json"


class ToolResultInputConfiguration(BaseModel):
    toolUseId: str
    type: TYPE = "TEXT"
    textInputConfiguration: TextInputConfiguration = TextInputConfiguration()


class ToolInputSchema(BaseModel):
    json_: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
        },
        alias="json",
    )


class ToolSpec(BaseModel):
    name: str
    description: str
    inputSchema: ToolInputSchema


class Tool(BaseModel):
    toolSpec: ToolSpec


class ToolConfiguration(BaseModel):
    tools: List[Tool]


class SessionStart(BaseModel):
    inferenceConfiguration: InferenceConfiguration


class InputTextContentStart(BaseModel):
    promptName: str
    contentName: str
    type: TYPE = "TEXT"
    interactive: bool = False
    role: ROLE
    textInputConfiguration: TextInputConfiguration


class InputAudioContentStart(BaseModel):
    promptName: str
    contentName: str
    type: TYPE = "AUDIO"
    interactive: bool = True
    role: ROLE = "USER"
    audioInputConfiguration: AudioInputConfiguration


class InputToolContentStart(BaseModel):
    promptName: str
    contentName: str
    type: TYPE = "TOOL"
    interactive: bool = False
    role: ROLE = "TOOL"
    toolResultInputConfiguration: ToolResultInputConfiguration


class PromptStart(BaseModel):
    promptName: str
    textOutputConfiguration: TextOutputConfiguration
    audioOutputConfiguration: AudioOutputConfiguration
    toolUseOutputConfiguration: ToolUseOutputConfiguration
    toolConfiguration: ToolConfiguration


class TextInput(BaseModel):
    promptName: str
    contentName: str
    content: str


class AudioInput(BaseModel):
    promptName: str
    contentName: str
    content: str


class ToolResult(BaseModel):
    promptName: str
    contentName: str
    content: str


class ContentEndEvent(BaseModel):
    promptName: str
    contentName: str


class PromptEnd(BaseModel):
    promptName: str


class SessionEnd(BaseModel):
    pass


class SessionStartEvent(BaseModel):
    sessionStart: SessionStart


class InputTextContentStartEvent(BaseModel):
    contentStart: InputTextContentStart


class InputAudioContentStartEvent(BaseModel):
    contentStart: InputAudioContentStart


class InputToolContentStartEvent(BaseModel):
    contentStart: InputToolContentStart


class PromptStartEvent(BaseModel):
    promptStart: PromptStart


class TextInputContentEvent(BaseModel):
    textInput: TextInput


class AudioInputContentEvent(BaseModel):
    audioInput: AudioInput


class ToolResultContentEvent(BaseModel):
    toolResult: ToolResult


class InputContentEndEvent(BaseModel):
    contentEnd: ContentEndEvent


class PromptEndEvent(BaseModel):
    promptEnd: PromptEnd


class SessionEndEvent(BaseModel):
    sessionEnd: SessionEnd


class Event(BaseModel):
    event: Union[
        SessionStartEvent,
        InputTextContentStartEvent,
        InputAudioContentStartEvent,
        InputToolContentStartEvent,
        PromptStartEvent,
        TextInputContentEvent,
        AudioInputContentEvent,
        ToolResultContentEvent,
        InputContentEndEvent,
        PromptEndEvent,
        SessionEndEvent,
    ]


class SonicEventBuilder:
    @staticmethod
    def create_session_start_event(
        max_tokens: int = 1024,
        top_p: float = 0.9,
        temperature: float = 0.7,
    ) -> str:
        event = Event(
            event=SessionStartEvent(
                sessionStart=SessionStart(
                    inferenceConfiguration=InferenceConfiguration(
                        maxTokens=max_tokens,
                        topP=top_p,
                        temperature=temperature,
                    )
                )
            )
        )
        return event.model_dump_json(exclude_none=False)

    @staticmethod
    def create_audio_content_start_event(
        prompt_name: str,
        content_name: str,
        sample_rate: SAMPLE_RATE_HERTZ = 16_000,
    ) -> str:
        event = Event(
            event=InputAudioContentStartEvent(
                contentStart=InputAudioContentStart(
                    promptName=prompt_name,
                    contentName=content_name,
                    audioInputConfiguration=AudioInputConfiguration(
                        sampleRateHertz=sample_rate,
                    ),
                )
            )
        )
        return event.model_dump_json(exclude_none=True)

    @staticmethod
    def create_text_content_start_event(
        prompt_name: str,
        content_name: str,
        role: ROLE,
    ) -> str:
        event = Event(
            event=InputTextContentStartEvent(
                contentStart=InputTextContentStart(
                    promptName=prompt_name,
                    contentName=content_name,
                    role=role,
                    textInputConfiguration=TextInputConfiguration(),
                )
            )
        )
        return event.model_dump_json(exclude_none=True)

    @staticmethod
    def create_tool_content_start_event(
        prompt_name: str,
        content_name: str,
        tool_use_id: str,
    ) -> str:
        event = Event(
            event=InputToolContentStartEvent(
                contentStart=InputToolContentStart(
                    promptName=prompt_name,
                    contentName=content_name,
                    toolResultInputConfiguration=ToolResultInputConfiguration(
                        toolUseId=tool_use_id,
                        textInputConfiguration=TextInputConfiguration(),
                    ),
                )
            )
        )
        return event.model_dump_json(exclude_none=True)

    @staticmethod
    def create_audio_input_event(
        prompt_name: str,
        content_name: str,
        audio_content: str,
    ) -> str:
        event = Event(
            event=AudioInputContentEvent(
                audioInput=AudioInput(
                    promptName=prompt_name,
                    contentName=content_name,
                    content=audio_content,
                )
            )
        )
        return event.model_dump_json(exclude_none=True)

    @staticmethod
    def create_text_input_event(
        prompt_name: str,
        content_name: str,
        content: str,
    ) -> str:
        event = Event(
            event=TextInputContentEvent(
                textInput=TextInput(
                    promptName=prompt_name,
                    contentName=content_name,
                    content=content,
                )
            )
        )
        return event.model_dump_json(exclude_none=True)

    @staticmethod
    def create_tool_result_event(
        prompt_name: str,
        content_name: str,
        content: Union[str, Dict[str, Any]],
    ) -> str:
        if isinstance(content, dict):
            content_str = json.dumps(content)
        else:
            content_str = content

        event = Event(
            event=ToolResultContentEvent(
                toolResult=ToolResult(
                    promptName=prompt_name,
                    contentName=content_name,
                    content=content_str,
                )
            )
        )
        return event.model_dump_json(exclude_none=True)

    @staticmethod
    def create_content_end_event(
        prompt_name: str,
        content_name: str,
    ) -> str:
        event = Event(
            event=InputContentEndEvent(
                contentEnd=ContentEndEvent(
                    promptName=prompt_name,
                    contentName=content_name,
                )
            )
        )
        return event.model_dump_json(exclude_none=True)

    @staticmethod
    def create_prompt_end_event(prompt_name: str) -> str:
        event = Event(
            event=PromptEndEvent(
                promptEnd=PromptEnd(promptName=prompt_name),
            )
        )
        return event.model_dump_json(exclude_none=True)

    @staticmethod
    def create_session_end_event() -> str:
        event = Event(
            event=SessionEndEvent(sessionEnd=SessionEnd()),
        )
        return event.model_dump_json(exclude_none=True)

    @staticmethod
    def create_prompt_start_event(
        prompt_name: str,
        voice_id: VOICE_ID,
        sample_rate: SAMPLE_RATE_HERTZ,
        tool_configuration: Optional[ToolConfiguration] = None,
    ) -> str:
        tool_configuration = tool_configuration or ToolConfiguration(tools=[])

        tool_objects = [
            Tool(
                toolSpec=ToolSpec(
                    name=tool.toolSpec.name,
                    description=tool.toolSpec.description,
                    inputSchema=ToolInputSchema(json=tool.toolSpec.inputSchema.json),
                )
            )
            for tool in tool_configuration.tools
        ]
        event = Event(
            event=PromptStartEvent(
                promptStart=PromptStart(
                    promptName=prompt_name,
                    textOutputConfiguration=TextOutputConfiguration(),
                    audioOutputConfiguration=AudioOutputConfiguration(
                        voiceId=voice_id, sampleRateHertz=sample_rate
                    ),
                    toolUseOutputConfiguration=ToolUseOutputConfiguration(),
                    toolConfiguration=ToolConfiguration(tools=tool_objects),
                )
            )
        )
        return event.model_dump_json(exclude_none=True)
