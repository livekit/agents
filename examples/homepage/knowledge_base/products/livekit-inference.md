LiveKit Inference, the unified API for STT, LLM, and TTS models without separate provider API keys, including supported models, pricing, and limits.

OVERVIEW:
LiveKit Inference is a built-in feature of LiveKit Cloud that gives you access to many of the best AI models for voice agents without needing separate API keys or accounts with each provider. It supports models from OpenAI, Google, Deepgram, AssemblyAI, Cartesia, ElevenLabs, Inworld, Rime, and more. LiveKit Inference covers three categories of models: large language models (LLM) for intelligence and reasoning, speech-to-text (STT) for transcription, and text-to-speech (TTS) for generating speech. It is included in every LiveKit Cloud plan, including the free Build plan.

KEY BENEFITS:
No API keys needed. You do not need to sign up for accounts with individual model providers or manage separate API keys. Everything is accessed through your LiveKit Cloud credentials. Unified billing means all model usage appears on a single LiveKit invoice instead of separate bills from each provider. LiveKit handles optimized routing and infrastructure so models are served with low latency. You can mix and match models from different providers freely. There are no additional plugins or dependencies to install beyond the core Agents SDK.

HOW IT WORKS WITH THE AGENTS SDK:
To use LiveKit Inference, import the inference module from the Agents SDK and use inference.STT, inference.LLM, and inference.TTS classes in your AgentSession. Models are specified using a provider slash model name format, like openai/gpt-4.1-mini for LLMs, deepgram/nova-3 for STT, or cartesia/sonic-3 for TTS. In Python, it looks like: from livekit.agents import AgentSession, inference, then create an AgentSession with stt equals inference.STT, llm equals inference.LLM, and tts equals inference.TTS. In Node.js, import AgentSession and inference from @livekit/agents and use new inference.STT, new inference.LLM, and new inference.TTS.

STRING DESCRIPTOR SHORTCUT:
As a convenience, you can pass model descriptor strings directly to the AgentSession instead of using inference classes. For example, you can pass "deepgram/nova-3:en" as the stt argument, "openai/gpt-4.1-mini" as the llm argument, and "cartesia/sonic-3:voice-id-here" as the tts argument. The colon separates the model name from additional parameters like language for STT or voice ID for TTS.

SUPPORTED LLM MODELS:
OpenAI models include GPT-4o, GPT-4o mini, GPT-4.1, GPT-4.1 mini, GPT-4.1 nano, GPT-5, GPT-5 mini, GPT-5 nano, GPT-5.1, GPT-5.2, and GPT OSS 120B. OpenAI models are provided by Azure and OpenAI. GPT OSS 120B is also available through Baseten, Groq, and Cerebras. Google Gemini models include Gemini 3 Pro, Gemini 3 Flash, Gemini 2.5 Pro, Gemini 2.5 Flash, and Gemini 2.5 Flash Lite. Kimi K2 Instruct is available through Baseten. DeepSeek V3 and DeepSeek V3.2 are available through Baseten.

SUPPORTED STT MODELS:
AssemblyAI offers Universal-3 Pro Streaming with 6 languages, Universal-Streaming for English only, and Universal-Streaming-Multilingual with 6 languages. Cartesia offers Ink Whisper supporting 100 languages. Deepgram offers Flux for English only, Nova-3 for 9 languages, Nova-3 Medical for English, Nova-2 for 33 languages, Nova-2 Medical for English, Nova-2 Conversational AI for English, and Nova-2 Phonecall for English. ElevenLabs offers Scribe V2 Realtime supporting 41 languages.

SUPPORTED TTS MODELS:
Cartesia offers sonic-3 with over 40 languages, sonic-2 and sonic-turbo with 15 languages each, and the original sonic model. Deepgram offers aura-2 for English and Spanish. ElevenLabs offers eleven_flash_v2 for English, eleven_flash_v2_5 for over 30 languages, eleven_turbo_v2 for English, eleven_turbo_v2_5 for over 30 languages, and eleven_multilingual_v2 for 28 languages. Inworld offers inworld-tts-1.5-max and inworld-tts-1.5-mini for 13 languages, and inworld-tts-1-max and inworld-tts-1 for 12 languages. Rime offers arcana for 9 languages and mistv2 for 4 languages.

COMPARISON TO PLUGINS:
The alternative to LiveKit Inference is using open source plugins. Plugins connect directly to each model provider's API. You need your own account and API key for each provider. Plugins are installed as optional dependencies, for example uv add livekit-agents with openai in Python or pnpm add @livekit/agents-plugin-openai in Node.js. Plugins give you access to a wider range of providers and some provider-specific features not yet available through Inference, such as the OpenAI Realtime API for speech-to-speech models. You can mix and match LiveKit Inference and plugins in the same agent. For example, you could use inference.STT and inference.TTS for speech models while using a plugin for your LLM, or vice versa.

PRICING AND BILLING:
LiveKit Inference billing is usage-based. The free Build plan includes $2.50 in monthly inference credits that can be used across any combination of models. Unused credits do not roll over. Projects on the free plan have a hard quota, and new requests fail after exceeding it. Discounted rates are available on the Scale plan, and custom rates are available on the Enterprise plan. The latest pricing for all individual models is available on the LiveKit Inference pricing page at livekit.io/pricing/inference.

QUOTAS AND LIMITS:
On the free Build plan, you get 5 concurrent STT connections, 5 concurrent TTS connections, 100 LLM requests per minute, and 600,000 LLM tokens per minute. STT and TTS use persistent WebSocket connections with concurrency limits. LLM uses stateless HTTP with rate limits for requests per minute and tokens per minute. Higher limits are available on paid plans. Scale plan customers can request limit increases in their project settings.

REGIONAL AVAILABILITY AND OPTIMIZED ROUTING:
LiveKit Inference runs within the LiveKit Cloud global infrastructure. LiveKit Cloud has a global edge network where users connect to the closest region for minimal latency. Some models have regional deployments for reduced latency. For example, Deepgram models have an integrated deployment in Mumbai, India, delivering significantly lower latency for voice agents serving users in that region. LiveKit handles routing automatically to minimize latency.

GETTING STARTED:
Sign up for a free LiveKit Cloud account. Set your LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET environment variables. Import the inference module from the Agents SDK and configure your AgentSession with inference.STT, inference.LLM, and inference.TTS. No other API keys or plugin installations are needed. Run your agent with the dev command for development or start for production.

LIVEKIT INFERENCE IS NOT AVAILABLE FOR SELF-HOSTED:
LiveKit Inference is a LiveKit Cloud feature only. If you self-host LiveKit, you must use model plugins instead and manage your own API keys and accounts with each provider. You would also need to remove the LiveKit Cloud noise cancellation plugin if self-hosting.
