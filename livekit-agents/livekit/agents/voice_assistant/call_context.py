_ContextVar = contextvars.ContextVar("voice_assistant_contextvar")


class AssistantCallContext:
    def __init__(self, assistant: "VoiceAssistant", llm_stream: allm.LLMStream) -> None:
        self._assistant = assistant
        self._metadata = dict()
        self._llm_stream = llm_stream

    @staticmethod
    def get_current() -> "AssistantCallContext":
        return _ContextVar.get()

    @property
    def assistant(self) -> "VoiceAssistant":
        return self._assistant

    def store_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def llm_stream(self) -> allm.LLMStream:
        return self._llm_stream
