# livekit-plugins-openai

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
