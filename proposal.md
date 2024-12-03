## Flexible PipelineAgent IO

Rename ``VoicePipelineAgent`` to ``PipelineAgent``

This API is non-breaking (backward compatible)

STT -> LLM -> TTS

```python
# rtc.AudioStream already satisfies this interface
AudioInput = AsyncIterator[rtc.AudioFrame | rtc.AudioFrameEvent]

# rtc.AudioSource already satisfies this interface
class AudioOutput(Protocol):
    async def capture_frame(self, frame: rtc.AudioFrame) -> None: ...

    def flush(self) -> None: ...

    def clear_queue(self) -> None: ...


class TextOutput(Protocol):
    async def write(self, text: str) -> None: ...

    def flush(self) -> None: ...


class PipelineIO(ABC):

    def before_stt_node(self, source: AsyncIterator[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
        return source

    def after_stt_node(self, source: AsyncIterator[SpeechEvent]) -> AsyncIterator[SpeechEvent]:
        return source

    def before_llm_node(self, chat_ctx: ChatContext) -> AsyncIterator[ChatChunk] | None:
        return None

    def after_llm_node(self, source: AsyncIterator[ChatChunk]) -> AsyncIterator[ChatChunk]:
        return source

    def before_tts_node(self, source: AsyncIterator[str] | str) -> AsyncIterator[rtc.AudioFrame] | None:
        return source

    def after_tts_node(self, source: AsyncIterator[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
        return source


STTFnc = Callable[[AsyncIterator[rtc.AudioFrame]], AsyncIterator[SpeechEvent]]
VADFnc = Callable[[AsyncIterator[rtc.AudioFrame]], AsyncIterator[VADEvent]]
LLMFnc = Callable[[ChatContext], AsyncIterator[ChatChunk]]
TTSFnc = Callable[[AsyncIterator[str] | str], AsyncIterator[rtc.AudioFrame]]


@dataclass
class PipelineOutput:
    audio: AudioOutput | None = None
    text: TextOutput | None = None

class PipelineAgent(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        llm: LLM | LLMFnc,
        io: PipelineIO | None = None,
        stt: stt.STT | STTFnc | None = None,
        vad: vad.VAD | VADFnc | None = None,
        tts: tts.TTS | TTSFnc | None = None,
        turn_detector: _TurnDetectorModel | None = None,
        output: PipelineOutput | None = None,
        chat_ctx: ChatContext | None = None,
        fnc_ctx: FunctionContext | None = None,
        allow_interruptions: bool = True,
        interrupt_speech_duration: float = 0.5,
        interrupt_min_words: int = 0,
        min_endpointing_delay: float = 0.5,
        max_nested_fnc_calls: int = 1,
        preemptive_synthesis: bool = False,
        transcription: AgentTranscriptionOptions = AgentTranscriptionOptions(),
        plotting: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:


# to avoid cumbersome code when you have both, before and after callbacks, users can define their
# STT/LLM/TTS nodes directly.

def my_llm(chat_ctx: ChatContext) -> AsyncIterator[ChatChunk]:
    http_session = aiohttp.ClientSession()
    try:
        yield from my_llm_impl(http_session, chat_ctx)
    finally:
        await http_session.close()


my_agent = PipelineAgent(llm=my_llm) # Internally this will convert my_llm to an actual LLM (e.g LLM.from_fnc(my_llm))

-> We should auto detect metrics when we can when nodes are defined as functions


# disable/enable audio/text output
my_agent.output.audio = None
my_agent.output.audio = rtc.AudioSource()

my_agent.output.text = None
my_agent.output.text = TextOutput()


# e.g: sync audio and text output for transcription (not detached from the VoicePipelineAgent)
class SyncedTranscription(AudioOutput, TextOutput):
    def __init__(self, *, text: TextOutput, audio: AudioOutput) -> None: # also has different outputs for passthroughts
        pass


tr_sync = SyncedTranscription()
my_agent.output.audio = tr_sync
my_agent.output.text = tr_sync


# e.g: speedup tts output
class MyIO(PipelineIO):
    async def after_tts_node(self, source: AsyncIterator[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
        yield from AudioSpeedProcessor(source, 2.0)

my_agent = PipelineAgent(llm=my_llm, io=MyIO())



e.g: modify llm output
class MyIO(PipelineIO):
    async def after_llm_node(self, source: AsyncIterator[ChatChunk]) -> AsyncIterator[ChatChunk]:
        async for chunk in source:
            chunk.text = chunk.text.upper()
            yield chunk

my_agent = PipelineAgent(llm=my_llm, io=MyIO())


```
