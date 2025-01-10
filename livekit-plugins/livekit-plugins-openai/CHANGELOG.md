# livekit-plugins-openai

## 0.10.14

### Patch Changes

- fix: revert from weakset to list in multimodal for maintaining sessions - [#1326](https://github.com/livekit/agents/pull/1326) ([@jayeshp19](https://github.com/jayeshp19))

## 0.10.13

### Patch Changes

- improved handling of LLM errors, do not retry if already began - [#1298](https://github.com/livekit/agents/pull/1298) ([@davidzhao](https://github.com/davidzhao))

- make multimodal class generic and support gemini live api - [#1240](https://github.com/livekit/agents/pull/1240) ([@jayeshp19](https://github.com/jayeshp19))

## 0.10.12

### Patch Changes

- fix unknown `metadata` & `store` fields on OpenAI-like API - [#1276](https://github.com/livekit/agents/pull/1276) ([@theomonnom](https://github.com/theomonnom))

## 0.10.11

### Patch Changes

- Moved create_ai_function_info to function_context.py for better reusability and reduce repetation - [#1260](https://github.com/livekit/agents/pull/1260) ([@jayeshp19](https://github.com/jayeshp19))

- add on_duplicate option for multimodal agent response create - [#1204](https://github.com/livekit/agents/pull/1204) ([@longcw](https://github.com/longcw))

- Add support for OpenAI's "detail" parameter to ChatImage - [#1213](https://github.com/livekit/agents/pull/1213) ([@bcherry](https://github.com/bcherry))

  Add support for data URLs on ChatImage in the Anthropic plugin.

- filter out empty message for set chat ctx in realtime model - [#1245](https://github.com/livekit/agents/pull/1245) ([@longcw](https://github.com/longcw))

- fix: correctly parse function argument types - [#1221](https://github.com/livekit/agents/pull/1221) ([@jayeshp19](https://github.com/jayeshp19))

- add session_updated event for RealtimeSession - [#1253](https://github.com/livekit/agents/pull/1253) ([@longcw](https://github.com/longcw))

- added llama 3.3 70b to model definitions - [#1233](https://github.com/livekit/agents/pull/1233) ([@davidzhao](https://github.com/davidzhao))

- update default realtime model to gpt-4o-realtime-preview-2024-12-17 - [#1250](https://github.com/livekit/agents/pull/1250) ([@davidzhao](https://github.com/davidzhao))

- Fix center_aspect_fit bug, add scale_aspect_fit and scale_aspect_fill resizing options. - [#1222](https://github.com/livekit/agents/pull/1222) ([@bcherry](https://github.com/bcherry))

  Make scale_aspect_fit the new default resizing option for video frames.

## 0.10.10

### Patch Changes

- add `google/gemini-2.0-flash-exp` as default model for vertex - [#1214](https://github.com/livekit/agents/pull/1214) ([@jayeshp19](https://github.com/jayeshp19))

- emit error event for realtime model - [#1200](https://github.com/livekit/agents/pull/1200) ([@longcw](https://github.com/longcw))

- fix: return structured output from func calls - [#1187](https://github.com/livekit/agents/pull/1187) ([@jayeshp19](https://github.com/jayeshp19))

- Handle optional func args in tool calls when set to `None` - [#1211](https://github.com/livekit/agents/pull/1211) ([@jayeshp19](https://github.com/jayeshp19))

- fix: openai llm retries - [#1196](https://github.com/livekit/agents/pull/1196) ([@theomonnom](https://github.com/theomonnom))

- Improvements to end of turn plugin, ensure STT language settings. - [#1195](https://github.com/livekit/agents/pull/1195) ([@davidzhao](https://github.com/davidzhao))

- fix: Handle optional func args in tool calls when set to `None` - [#1211](https://github.com/livekit/agents/pull/1211) ([@jayeshp19](https://github.com/jayeshp19))

## 0.10.9

### Patch Changes

- fix release - [#1176](https://github.com/livekit/agents/pull/1176) ([@theomonnom](https://github.com/theomonnom))

## 0.10.8

### Patch Changes

- fix uncatched OAI errors - [#1158](https://github.com/livekit/agents/pull/1158) ([@theomonnom](https://github.com/theomonnom))

- feat: stt retry & stt.FallbackAdapter - [#1114](https://github.com/livekit/agents/pull/1114) ([@theomonnom](https://github.com/theomonnom))

- project id fix for google - [#1115](https://github.com/livekit/agents/pull/1115) ([@jayeshp19](https://github.com/jayeshp19))

- Add retries to recover from text mode to audio model for realtime API - [#1121](https://github.com/livekit/agents/pull/1121) ([@longcw](https://github.com/longcw))

- Support for Python 3.13, relaxed Pillow version requirement for 10.x - [#1127](https://github.com/livekit/agents/pull/1127) ([@davidzhao](https://github.com/davidzhao))

- support for custom tool use in LLMs - [#1102](https://github.com/livekit/agents/pull/1102) ([@jayeshp19](https://github.com/jayeshp19))

- feat: tts retry & tts.FallbackAdapter - [#1074](https://github.com/livekit/agents/pull/1074) ([@theomonnom](https://github.com/theomonnom))

- Add new OpenAI realtime voices - [#1116](https://github.com/livekit/agents/pull/1116) ([@bcherry](https://github.com/bcherry))

- Expose multimodal agent metrics - [#1080](https://github.com/livekit/agents/pull/1080) ([@longcw](https://github.com/longcw))

- feat: llm retry & llm.FallbackAdapter - [#1132](https://github.com/livekit/agents/pull/1132) ([@theomonnom](https://github.com/theomonnom))

- vertex ai support with openai library - [#1084](https://github.com/livekit/agents/pull/1084) ([@jayeshp19](https://github.com/jayeshp19))

## 0.10.7

### Patch Changes

- fix realtime API audio format values - [#1092](https://github.com/livekit/agents/pull/1092) ([@longcw](https://github.com/longcw))

- make ConversationItem.create and delete return a Future in Realtime model - [#1085](https://github.com/livekit/agents/pull/1085) ([@longcw](https://github.com/longcw))

## 0.10.6

### Patch Changes

- Expose usage metrics for Realtime model - [#1036](https://github.com/livekit/agents/pull/1036) ([@yuyuma](https://github.com/yuyuma))

- sync the Realtime API converstation items and add set_chat_ctx - [#1015](https://github.com/livekit/agents/pull/1015) ([@longcw](https://github.com/longcw))

## 0.10.5

### Patch Changes

- fix: Azure realtime model does not accept null for max_response_output_tokens - [#927](https://github.com/livekit/agents/pull/927) ([@davidzhao](https://github.com/davidzhao))

- add update_options to TTS - [#922](https://github.com/livekit/agents/pull/922) ([@theomonnom](https://github.com/theomonnom))

- Groq integration with Whisper-compatible STT endpoints - [#986](https://github.com/livekit/agents/pull/986) ([@jayeshp19](https://github.com/jayeshp19))

- pipelineagent: expose timing metrics & api errors wip - [#957](https://github.com/livekit/agents/pull/957) ([@theomonnom](https://github.com/theomonnom))

- openai: fix low timeouts - [#926](https://github.com/livekit/agents/pull/926) ([@theomonnom](https://github.com/theomonnom))

- expose usage metrics - [#984](https://github.com/livekit/agents/pull/984) ([@theomonnom](https://github.com/theomonnom))

## 0.10.4

### Patch Changes

- add x.ai support - [#907](https://github.com/livekit/agents/pull/907) ([@theomonnom](https://github.com/theomonnom))

- Fix functions to include content - [#897](https://github.com/livekit/agents/pull/897) ([@martin-purplefish](https://github.com/martin-purplefish))

## 0.10.3

### Patch Changes

- fix: handle when STT does not return any speech - [#854](https://github.com/livekit/agents/pull/854) ([@davidzhao](https://github.com/davidzhao))

- Support for Realtime API with Azure OpenAI - [#848](https://github.com/livekit/agents/pull/848) ([@davidzhao](https://github.com/davidzhao))

## 0.10.2

### Patch Changes

- oai-realtime: fix function calls - [#826](https://github.com/livekit/agents/pull/826) ([@KillianLucas](https://github.com/KillianLucas))

## 0.10.1

### Patch Changes

- oai-realtime: log response errors - [#819](https://github.com/livekit/agents/pull/819) ([@theomonnom](https://github.com/theomonnom))

## 0.10.0

### Minor Changes

- OpenAI Realtime API support - [#814](https://github.com/livekit/agents/pull/814) ([@theomonnom](https://github.com/theomonnom))

### Patch Changes

- Add Telnyx integration for LLM - [#803](https://github.com/livekit/agents/pull/803) ([@jamestwhedbee](https://github.com/jamestwhedbee))

## 0.8.5

### Patch Changes

- Fix function for OpenAI Assistants - [#784](https://github.com/livekit/agents/pull/784) ([@keepingitneil](https://github.com/keepingitneil))

## 0.8.4

### Patch Changes

- avoid returning tiny frames from TTS - [#747](https://github.com/livekit/agents/pull/747) ([@theomonnom](https://github.com/theomonnom))

- Fixing Assistant API Vision Capabilities - [#771](https://github.com/livekit/agents/pull/771) ([@keepingitneil](https://github.com/keepingitneil))

## 0.8.3

### Patch Changes

- Introduce function calling to OpenAI Assistants - [#710](https://github.com/livekit/agents/pull/710) ([@keepingitneil](https://github.com/keepingitneil))

- Add Cerebras to OpenAI Plugin - [#731](https://github.com/livekit/agents/pull/731) ([@henrytwo](https://github.com/henrytwo))

## 0.8.2

### Patch Changes

- Add deepseek LLMs at OpenAI plugin - [#714](https://github.com/livekit/agents/pull/714) ([@lenage](https://github.com/lenage))

- skip processing of choice.delta when it is None - [#705](https://github.com/livekit/agents/pull/705) ([@theomonnom](https://github.com/theomonnom))

## 0.8.1

### Patch Changes

- add support for Ollama, Perplexity, Fireworks, Octo, Together, and Groq LLMs through the OpenAI API - [#611](https://github.com/livekit/agents/pull/611) ([@nbsp](https://github.com/nbsp))

- allow sending user IDs - [#633](https://github.com/livekit/agents/pull/633) ([@nbsp](https://github.com/nbsp))

- Support OpenAI Assistants API as a beta feature under `livekit.plugins.openai.beta` - [#601](https://github.com/livekit/agents/pull/601) ([@keepingitneil](https://github.com/keepingitneil))
  Add \_metadata to ChatCtx and ChatMessage which can be used (in the case of OpenAI assistants) for bookeeping to sync local state with remote, OpenAI state

## 0.8.0

### Minor Changes

- openai: use openai client for stt - [#583](https://github.com/livekit/agents/pull/583) ([@theomonnom](https://github.com/theomonnom))

### Patch Changes

- openai: add api_key argument - [#580](https://github.com/livekit/agents/pull/580) ([@theomonnom](https://github.com/theomonnom))

- openai: fix incorrect API urls on Windows - [#575](https://github.com/livekit/agents/pull/575) ([@theomonnom](https://github.com/theomonnom))

## 0.7.1

### Patch Changes

- set timeout to 5 seconds - [#524](https://github.com/livekit/agents/pull/524) ([@nbsp](https://github.com/nbsp))

## 0.7.0

### Minor Changes

- dev prerelease - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

### Patch Changes

- test release - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

- pull: '--rebase --autostash ...' - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

- Default loglevel to warn - [#472](https://github.com/livekit/agents/pull/472) ([@lukasIO](https://github.com/lukasIO))

- bump versions to update dependencies - [#510](https://github.com/livekit/agents/pull/510) ([@theomonnom](https://github.com/theomonnom))

- test release - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

- fix changesets release CI - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

- release v0.8.0 - [`6e74aa714c2dfaa8212db4528d7b59d095b6c660`](https://github.com/livekit/agents/commit/6e74aa714c2dfaa8212db4528d7b59d095b6c660) ([@theomonnom](https://github.com/theomonnom))

- dev fixes - multiprocessing & voiceassistant - [#493](https://github.com/livekit/agents/pull/493) ([@theomonnom](https://github.com/theomonnom))

## 0.7.0-dev.7

### Patch Changes

- pull: '--rebase --autostash ...' - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.7.0-dev.6

### Patch Changes

- test release - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.7.0-dev.5

### Patch Changes

- test release - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.7.0-dev.4

### Patch Changes

- fix changesets release CI - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.7.0-dev.3

### Patch Changes

- bump versions to update dependencies - [#510](https://github.com/livekit/agents/pull/510) ([@theomonnom](https://github.com/theomonnom))

## 0.7.0-dev.2

### Patch Changes

- dev fixes - multiprocessing & voiceassistant - [#493](https://github.com/livekit/agents/pull/493) ([@theomonnom](https://github.com/theomonnom))

## 0.7.0-dev.1

### Minor Changes

- dev prerelease - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.6.1-dev.0

### Patch Changes

- Default loglevel to warn - [#472](https://github.com/livekit/agents/pull/472) ([@lukasIO](https://github.com/lukasIO))
