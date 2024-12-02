## Flexible PipelineAgent IO

Rename ``VoicePipelineAgent`` to ``PipelineAgent``

STT -> LLM -> TTS

```python
# rtc.AudioStream already satisfies this interface
class AudioInput(Protocol):
    async def __anext__(self) -> rtc.AudioFrame | rtc.AudioFrameEvent: ...


# rtc.AudioSource already satisfies this interface
class AudioOutput(Protocol):
    async def capture_frame(self, frame: rtc.AudioFrame) -> None: ...

    async def wait_for_playout(self) -> None: ...

    def clear_queue(self) -> None: ...


class PipelineIO(ABC):

    def before_stt_node(self, source: AsyncIterator[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
        return source

    def after_stt_node(self, source: AsyncIterator[SpeechEvent]) -> AsyncIterator[SpeechEvent]:
        return source

    def before_llm_node(self, chat_ctx: ChatContext) -> AsyncIterator[ChatChunk] | None:
        return None

    def after_llm_node(self, source: AsyncIterator[ChatChunk]) -> AsyncIterator[ChatChunk]:
        return source

    def before_tts_node(self, source: AsyncIterator[str]) -> AsyncIterator[rtc.AudioFrame] | None:
        return source

    def after_tts_node(self, source: AsyncIterator[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
        return source

agent = PipelineAgent(
    stt=STT(), # optional
    llm=LLM(),
    tts=TTS(), # optional
    vad=VAD(), # optional
    eou=TurnDetector(), # optional
    io=MyPipelineIO(),
    audio_input_enabled=True,
    audio_output_enabled=True,
)


# to avoid cumbersome code when you have both, before and after callbacks, users can define their
# STT/LLM/TTS nodes directly.



```
