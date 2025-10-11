"""豆包 Realtime API 事件定义"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal


class DoubaoEventID(IntEnum):
    """豆包事件ID"""
    # Connect类事件
    START_CONNECTION = 1
    FINISH_CONNECTION = 2
    CONNECTION_STARTED = 50
    CONNECTION_FAILED = 51
    CONNECTION_FINISHED = 52

    # Session类事件
    START_SESSION = 100
    FINISH_SESSION = 102
    SESSION_STARTED = 150
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    USAGE_RESPONSE = 154

    # 客户端任务事件
    TASK_REQUEST = 200  # 客户端上传音频
    SAY_HELLO = 300  # 客户端提交打招呼文本
    CHAT_TTS_TEXT = 500  # 指定文本合成音频
    CHAT_TEXT_QUERY = 501  # 用户输入文本query
    CHAT_RAG_TEXT = 502  # 输入外部RAG知识

    # 服务端TTS事件
    TTS_SENTENCE_START = 350
    TTS_SENTENCE_END = 351
    TTS_RESPONSE = 352  # 返回音频数据
    TTS_ENDED = 359

    # 服务端ASR事件
    ASR_INFO = 450  # 识别出首字（用于打断）
    ASR_RESPONSE = 451  # 识别出文本内容
    ASR_ENDED = 459

    # 服务端聊天事件
    CHAT_RESPONSE = 550  # 模型回复文本
    CHAT_ENDED = 559


@dataclass
class LocationInfo:
    """位置信息"""
    longitude: float | None = None
    latitude: float | None = None
    city: str | None = None
    country: str | None = "中国"
    province: str | None = None
    district: str | None = None
    town: str | None = None
    country_code: str | None = "CN"
    address: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {}
        if self.longitude is not None:
            result["longitude"] = self.longitude
        if self.latitude is not None:
            result["latitude"] = self.latitude
        if self.city:
            result["city"] = self.city
        if self.country:
            result["country"] = self.country
        if self.province:
            result["province"] = self.province
        if self.district:
            result["district"] = self.district
        if self.town:
            result["town"] = self.town
        if self.country_code:
            result["country_code"] = self.country_code
        if self.address:
            result["address"] = self.address
        return result


@dataclass
class ASRConfig:
    """ASR配置"""
    end_smooth_window_ms: int = 1500  # 判断用户停止说话的时间，默认1500ms
    enable_custom_vad: bool = False  # 是否开启自定义VAD参数

    def to_dict(self) -> dict[str, Any]:
        return {
            "extra": {
                "end_smooth_window_ms": self.end_smooth_window_ms,
                "enable_custom_vad": self.enable_custom_vad,
            }
        }


@dataclass
class DialogConfig:
    """对话配置"""
    bot_name: str | None = "豆包"  # 人设名称，最长20字符
    system_role: str | None = None  # 背景人设信息
    speaking_style: str | None = None  # 对话风格
    dialog_id: str | None = None  # 对话ID，用于上下文记忆
    character_manifest: str | None = None  # 角色描述（SC版本）
    location: LocationInfo | None = None  # 用户位置信息
    strict_audit: bool = True  # 安全审核等级
    audit_response: str | None = None  # 命中审核后的回复话术
    enable_volc_websearch: bool = False  # 是否开启内置联网
    volc_websearch_type: str = "web_summary"  # 搜索类型: web_summary/web
    volc_websearch_api_key: str | None = None  # 搜索API密钥
    volc_websearch_result_count: int = 10  # 搜索结果条数
    volc_websearch_no_result_message: str | None = None  # 无结果回复话术
    input_mod: str | None = None  # 输入模式: text/audio_file/麦克风（默认）
    model: str = "O"  # 模型版本: O/SC

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if self.bot_name:
            result["bot_name"] = self.bot_name
        if self.system_role:
            result["system_role"] = self.system_role
        if self.speaking_style:
            result["speaking_style"] = self.speaking_style
        if self.dialog_id:
            result["dialog_id"] = self.dialog_id
        if self.character_manifest:
            result["character_manifest"] = self.character_manifest
        if self.location:
            result["location"] = self.location.to_dict()

        extra: dict[str, Any] = {}
        extra["strict_audit"] = self.strict_audit
        if self.audit_response:
            extra["audit_response"] = self.audit_response
        if self.enable_volc_websearch:
            extra["enable_volc_websearch"] = self.enable_volc_websearch
            extra["volc_websearch_type"] = self.volc_websearch_type
            if self.volc_websearch_api_key:
                extra["volc_websearch_api_key"] = self.volc_websearch_api_key
            extra["volc_websearch_result_count"] = self.volc_websearch_result_count
            if self.volc_websearch_no_result_message:
                extra["volc_websearch_no_result_message"] = self.volc_websearch_no_result_message
        if self.input_mod:
            extra["input_mod"] = self.input_mod
        extra["model"] = self.model

        result["extra"] = extra
        return result


@dataclass
class TTSConfig:
    """TTS配置"""
    speaker: str = "zh_female_vv_jupiter_bigtts"  # 发音人
    audio_format: Literal["ogg_opus", "pcm", "pcm_s16le"] = "pcm"  # 音频格式
    sample_rate: int = 24000  # 采样率
    channel: int = 1  # 声道数

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "speaker": self.speaker,
        }

        if self.audio_format != "ogg_opus":
            result["audio_config"] = {
                "format": self.audio_format,
                "sample_rate": self.sample_rate,
                "channel": self.channel,
            }

        return result


# 客户端事件数据类

@dataclass
class StartConnectionEvent:
    """建立连接事件"""

    def to_dict(self) -> dict[str, Any]:
        return {}


@dataclass
class StartSessionEvent:
    """启动会话事件"""
    asr: ASRConfig | None = None
    dialog: DialogConfig | None = None
    tts: TTSConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if self.asr:
            result["asr"] = self.asr.to_dict()
        if self.dialog:
            result["dialog"] = self.dialog.to_dict()
        if self.tts:
            result["tts"] = self.tts.to_dict()

        return result


@dataclass
class SayHelloEvent:
    """打招呼事件"""
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content}


@dataclass
class ChatTTSTextEvent:
    """文本合成音频事件（流式）"""
    start: bool = False
    content: str = ""
    end: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "content": self.content,
            "end": self.end,
        }


@dataclass
class ChatTextQueryEvent:
    """文本query事件"""
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content}


@dataclass
class ChatRAGTextEvent:
    """外部RAG输入事件"""
    external_rag: str  # JSON数组字符串

    def to_dict(self) -> dict[str, Any]:
        return {"external_rag": self.external_rag}


# 服务端事件数据类

@dataclass
class SessionStartedEvent:
    """会话启动成功事件"""
    dialog_id: str


@dataclass
class TTSSentenceStartEvent:
    """TTS句子开始事件"""
    tts_type: str  # audit_content_risky/chat_tts_text/network/external_rag/default
    text: str


@dataclass
class ASRResponseEvent:
    """ASR识别结果事件"""
    results: list[dict[str, Any]]  # [{"text": str, "is_interim": bool}]


@dataclass
class ChatResponseEvent:
    """聊天回复事件"""
    content: str


@dataclass
class UsageResponseEvent:
    """用量信息事件"""
    usage: dict[str, int]  # input_text_tokens, input_audio_tokens, etc.


@dataclass
class ErrorEvent:
    """错误事件"""
    error: str