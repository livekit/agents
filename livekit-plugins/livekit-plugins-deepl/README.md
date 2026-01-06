# DeepL Translation plugin for LiveKit Agents

This plugin provides real-time text translation capabilities using [DeepL](https://www.deepl.com/) for LiveKit Agents.

## Installation

```bash
pip install livekit-plugins-deepl
```

## Pre-requisites

You'll need an API key from DeepL. It must be set as an environment variable: `DEEPL_AUTH_KEY`.

Optionally, you can specify the DeepL API endpoint using `DEEPL_SERVER_URL` to choose between Free and Pro API tiers:
*   For DeepL Free API: `export DEEPL_SERVER_URL="https://api-free.deepl.com"`
*   For DeepL Pro API: `export DEEPL_SERVER_URL="https://api.deepl.com"`

## Features

*   **Real-time Text Translation:** Translate text efficiently between a wide range of source and target languages supported by DeepL.
*   **Automatic Source Language Detection:** Supports DeepL's automatic source language detection.
*   **Flexible Integration:** Designed to integrate as a middleware step within your agent's conversational flow (e.g., STT output -> Translation -> LLM input or LLM output -> Translation -> TTS input).

## Supported Languages

DeepL supports a broad range of languages. For the most up-to-date list of supported source and target languages, please refer to the official [DeepL API documentation](https://developers.deepl.com/docs/getting-started/supported-languages).

## Example Usage

### Basic Translation

```python
import asyncio
from dotenv import load_dotenv

load_dotenv()

from livekit.plugins.deepl.translator import DeepLTranslationPlugin

async def translate_example():
    # Instantiate the plugin (this also registers it with the LiveKit Agents system)
    deepl_plugin = DeepLTranslationPlugin()

    source_text = "The quick brown fox jumps over the lazy dog."
    source_lang = "en" # or None for automatic detection
    target_lang = "fr"

    try:
        translated_text_list = await deepl_plugin.translate_text(
            text=source_text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        translated_text = translated_text_list[0] if translated_text_list else ""
        print(f"Original ({source_lang}): {source_text}")
        print(f"Translated ({target_lang}): {translated_text}")

        source_text_de = "Guten Tag, wie geht es Ihnen?"
        source_lang_de = "de"
        target_lang_de = "es"
        translated_text_es_list = await deepl_plugin.translate_text(
            text=source_text_de,
            source_lang=source_lang_de,
            target_lang=target_lang_de
        )
        translated_text_es = translated_text_es_list[0] if translated_text_es_list else ""
        print(f"Original ({source_lang_de}): {source_text_de}")
        print(f"Translated ({target_lang_de}): {translated_text_es}")

    except Exception as e:
        print(f"Translation error: {e}")

if __name__ == "__main__":
    asyncio.run(translate_example())
```

### Using with LiveKit Agents Framework (Conceptual Integration)

This plugin integrates as a service that your agent calls. It doesn't replace an STT, TTS, or LLM directly, but rather augments them.

```python
from dotenv import load_dotenv

load_dotenv()

import logging  
from typing import AsyncIterable

from livekit.agents import (  
    Agent,
    AgentServer,
    AgentSession,  
    JobContext,  
    JobProcess,
    cli,
    llm,  
    ModelSettings, 
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import deepgram, cartesia, silero  
from livekit.plugins.deepl import DeepLTranslationPlugin

logger = logging.getLogger(__name__)

class MultilingualAgent(Agent):
    def __init__(self, agent_language: str, user_language: str = None) -> None:  
        super().__init__(  
            instructions="You are a helpful voice assistant. Respond in the language you receive the input.",  
        )  
        self.user_language = user_language  
        self.agent_language = agent_language  
        self.translator = DeepLTranslationPlugin()

    async def on_enter(self):
        greeting = "Hello! I can help you in multiple languages. How can I assist you today?"
        greeting_list = await self.translator.translate_text(  
            text=greeting,  
            source_lang="en",  
            target_lang=self.agent_language  
        )
        await self.session.generate_reply(  
            instructions=greeting_list[0] if greeting_list else greeting,
            allow_interruptions=False
        )

    async def on_user_turn_completed(  
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage  
    ) -> None:  
        """Translate user input to agent's language before processing"""  
        if new_message.text_content and self.user_language != self.agent_language:    
            try:  
                # Translate user input to agent's operating language  
                translated_content = await self.translator.translate_text(  
                    text=new_message.text_content,  
                    source_lang=self.user_language,  
                    target_lang=self.agent_language  
                )  
                new_message.content = translated_content  
                logger.info(f"Translated from {self.user_language}: {translated_content}")  
            except Exception as e:  
                logger.error(f"Translation failed: {e}")  
                # Continue with original text if translation fails  
      
    async def llm_node(  
        self,   
        chat_ctx: llm.ChatContext,   
        tools: list[llm.Tool],   
        model_settings: ModelSettings  
    ) -> AsyncIterable[llm.ChatChunk | str]:  
        """Override to translate LLM output back to user's language"""  
        # Get the original LLM response  
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):  
            if isinstance(chunk, str):  
                # Translate the response back to user's language  
                try:  
                    translated_list = await self.translator.translate_text(  
                        text=chunk,  
                        source_lang=self.agent_language,
                        target_lang=self.user_language  
                    )  
                    yield translated_list[0] if translated_list else chunk  
                except Exception as e:  
                    logger.error(f"Output translation failed: {e}")  
                    yield chunk  # Fallback to original  
            else:  
                yield chunk
                
server = AgentServer()  

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session()
async def entrypoint(ctx: JobContext):  
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    agent_lang = "FR"  # Agent operates in French
    
    # Auto-detect user language
    agent = MultilingualAgent(
        agent_language=agent_lang
    )
    
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
    )
  
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(server)
```