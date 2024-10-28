# LiveKit Plugins Dify

To use:

```
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model=dg_model),
        llm=dify.LLM(
            base_url="Your dify API",
            api_key="Your Dify App API Key",
        ),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
    )
```

Notes: Do not use `initial_ctx = llm.ChatContext().append` to add system prompt or context. Implement these in your Dify.

Only 'Chatflow' is supported.