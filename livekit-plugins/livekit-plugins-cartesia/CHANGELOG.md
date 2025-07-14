# livekit-plugins-cartesia

## 0.4.12

### Patch Changes

- update to livekit python 1.0 - [`32e129ff1a4c3d28f363f4f2b2a355e29c8fe64d`](https://github.com/livekit/agents/commit/32e129ff1a4c3d28f363f4f2b2a355e29c8fe64d) ([@davidzhao](https://github.com/davidzhao))

## 0.4.11

### Patch Changes

- Add string type support to model parameter - [#1657](https://github.com/livekit/agents/pull/1657) ([@jayeshp19](https://github.com/jayeshp19))

## 0.4.10

### Patch Changes

- Adding new model literals, updating default to sonic-2 - [#1627](https://github.com/livekit/agents/pull/1627) ([@longcw](https://github.com/longcw))

## 0.4.9

### Patch Changes

- use streaming AudioDecoder to handle compressed encoding - [#1584](https://github.com/livekit/agents/pull/1584) ([@davidzhao](https://github.com/davidzhao))

- added a tts.prewarm method to start the connection pool early. - [#1587](https://github.com/livekit/agents/pull/1587) ([@davidzhao](https://github.com/davidzhao))

- update pool configuration for deepgram and cartesia - [#1605](https://github.com/livekit/agents/pull/1605) ([@jayeshp19](https://github.com/jayeshp19))

- updated livekit-agent reference to <1.0 - [#1607](https://github.com/livekit/agents/pull/1607) ([@davidzhao](https://github.com/davidzhao))

## 0.4.8

### Patch Changes

- feat: connection pooling. speeds up generation with STT/TTS providers - [#1538](https://github.com/livekit/agents/pull/1538) ([@davidzhao](https://github.com/davidzhao))

- remove update options from tts synthesis stream - [#1546](https://github.com/livekit/agents/pull/1546) ([@jayeshp19](https://github.com/jayeshp19))

## 0.4.7

### Patch Changes

- improved TTFB metrics for streaming TTS - [#1431](https://github.com/livekit/agents/pull/1431) ([@davidzhao](https://github.com/davidzhao))

## 0.4.6

### Patch Changes

- update Cartesia plugin default model and voice id - [#1346](https://github.com/livekit/agents/pull/1346) ([@noahlt](https://github.com/noahlt))

## 0.4.5

### Patch Changes

- fix release - [#1176](https://github.com/livekit/agents/pull/1176) ([@theomonnom](https://github.com/theomonnom))

## 0.4.4

### Patch Changes

- feat: tts retry & tts.FallbackAdapter - [#1074](https://github.com/livekit/agents/pull/1074) ([@theomonnom](https://github.com/theomonnom))

## 0.4.3

### Patch Changes

- add update_options to TTS - [#922](https://github.com/livekit/agents/pull/922) ([@theomonnom](https://github.com/theomonnom))

- pipelineagent: expose timing metrics & api errors wip - [#957](https://github.com/livekit/agents/pull/957) ([@theomonnom](https://github.com/theomonnom))

- expose usage metrics - [#984](https://github.com/livekit/agents/pull/984) ([@theomonnom](https://github.com/theomonnom))

## 0.4.2

### Patch Changes

- Add support for cartesia voice control - [#740](https://github.com/livekit/agents/pull/740) ([@bcherry](https://github.com/bcherry))

## 0.4.1

### Patch Changes

- Switch Cartesia to a sentence tokenizer and keep the same context id throughout. - [#608](https://github.com/livekit/agents/pull/608) ([@keepingitneil](https://github.com/keepingitneil))
  Propagate segment_id through the basic sentence tokenizer

## 0.3.0

### Minor Changes

- cartesia: correctly add spaces & fix tests - [#591](https://github.com/livekit/agents/pull/591) ([@theomonnom](https://github.com/theomonnom))

### Patch Changes

- fix log warnings & cartesia end of speech - [#603](https://github.com/livekit/agents/pull/603) ([@theomonnom](https://github.com/theomonnom))

- stt/tts: fix unread inputs when the input channel is closed - [#594](https://github.com/livekit/agents/pull/594) ([@theomonnom](https://github.com/theomonnom))

- Adds websockets streaming to Cartesia plugin - [#544](https://github.com/livekit/agents/pull/544) ([@sauhardjain](https://github.com/sauhardjain))

## 0.2.0

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

## 0.2.0-dev.7

### Patch Changes

- pull: '--rebase --autostash ...' - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.2.0-dev.6

### Patch Changes

- test release - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.2.0-dev.5

### Patch Changes

- test release - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.2.0-dev.4

### Patch Changes

- fix changesets release CI - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.2.0-dev.3

### Patch Changes

- bump versions to update dependencies - [#510](https://github.com/livekit/agents/pull/510) ([@theomonnom](https://github.com/theomonnom))

## 0.2.0-dev.2

### Patch Changes

- dev fixes - multiprocessing & voiceassistant - [#493](https://github.com/livekit/agents/pull/493) ([@theomonnom](https://github.com/theomonnom))

## 0.2.0-dev.1

### Minor Changes

- dev prerelease - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.1.2-dev.0

### Patch Changes

- Default loglevel to warn - [#472](https://github.com/livekit/agents/pull/472) ([@lukasIO](https://github.com/lukasIO))
