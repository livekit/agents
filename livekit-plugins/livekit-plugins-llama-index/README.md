# LiveKit Plugins Llama Index

Agent Framework plugin for using Llama Index. Currently supports [Query Engine](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/) and [Chat Engine](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/).

## Install

```bash
pip install livekit-plugins-llama-index
```

## Query Engine

Query Engine is primarily used for RAG. See [example voice agent](https://github.com/livekit/agents/blob/main/examples/voice-pipeline-agent/llamaindex-rag/query_engine.py)

## Chat Engine

Chat Engine can be used as an LLM within the framework.

```python
# load the existing index
storage_context = StorageContext.from_defaults(persist_dir=<mydir>)
index = load_index_from_storage(storage_context)

async def entrypoint(ctx: JobContext):
    ...
    chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT)
    assistant = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=llama_index.LLM(chat_engine=chat_engine),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
    )
```

full example [here](https://github.com/livekit/agents/blob/main/examples/voice-pipeline-agent/llamaindex-rag/chat_engine.py)
