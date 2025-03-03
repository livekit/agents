# livekit-plugins-deepgram

## 0.6.20

### Patch Changes

- fix(deepgram): fix STT keyterm parameter - [#1535](https://github.com/livekit/agents/pull/1535) ([@wdhwg001](https://github.com/wdhwg001))

- use connection pool for deepgram tts - [#1523](https://github.com/livekit/agents/pull/1523) ([@jayeshp19](https://github.com/jayeshp19))

- remove update options from tts synthesis stream - [#1546](https://github.com/livekit/agents/pull/1546) ([@jayeshp19](https://github.com/jayeshp19))

## 0.6.19

### Patch Changes

- deepgram: support for Nova-3 keyterms - [#1484](https://github.com/livekit/agents/pull/1484) ([@davidzhao](https://github.com/davidzhao))

## 0.6.18

### Patch Changes

- chore(Deepgram STT): add nova-3 model to type literal - [#1464](https://github.com/livekit/agents/pull/1464) ([@chasemcdo](https://github.com/chasemcdo))

## 0.6.17

### Patch Changes

- improved TTFB metrics for streaming TTS - [#1431](https://github.com/livekit/agents/pull/1431) ([@davidzhao](https://github.com/davidzhao))

## 0.6.16

### Patch Changes

- fix: Ensure STT exceptions are being propagated - [#1291](https://github.com/livekit/agents/pull/1291) ([@davidzhao](https://github.com/davidzhao))

## 0.6.15

### Patch Changes

- added streaming audio decoder for compressed audio. - [#1236](https://github.com/livekit/agents/pull/1236) ([@davidzhao](https://github.com/davidzhao))

- Support Deepgram TTS - [#1201](https://github.com/livekit/agents/pull/1201) ([@jayeshp19](https://github.com/jayeshp19))

## 0.6.14

### Patch Changes

- enable deepgram filler words by default to improve end of turn accuracy - [#1190](https://github.com/livekit/agents/pull/1190) ([@davidzhao](https://github.com/davidzhao))

## 0.6.13

### Patch Changes

- fix release - [#1176](https://github.com/livekit/agents/pull/1176) ([@theomonnom](https://github.com/theomonnom))

## 0.6.12

### Patch Changes

- feat: stt retry & stt.FallbackAdapter - [#1114](https://github.com/livekit/agents/pull/1114) ([@theomonnom](https://github.com/theomonnom))

- Added support for custom deepgram base url - [#1137](https://github.com/livekit/agents/pull/1137) ([@theomonnom](https://github.com/theomonnom))

## 0.6.11

### Patch Changes

- add PeriodicCollector utility for metrics - [#1094](https://github.com/livekit/agents/pull/1094) ([@davidzhao](https://github.com/davidzhao))

## 0.6.10

### Patch Changes

- fix Deepgram missing first word, disabled energy filter by default - [#1090](https://github.com/livekit/agents/pull/1090) ([@davidzhao](https://github.com/davidzhao))

## 0.6.9

### Patch Changes

- stt: reduce bandwidth usage by reducing sample_rate to 16khz - [#920](https://github.com/livekit/agents/pull/920) ([@theomonnom](https://github.com/theomonnom))

- deepgram: send finalize each time we stop sending audio - [#1004](https://github.com/livekit/agents/pull/1004) ([@theomonnom](https://github.com/theomonnom))

- pipelineagent: expose timing metrics & api errors wip - [#957](https://github.com/livekit/agents/pull/957) ([@theomonnom](https://github.com/theomonnom))

- expose usage metrics - [#984](https://github.com/livekit/agents/pull/984) ([@theomonnom](https://github.com/theomonnom))

## 0.6.8

### Patch Changes

- accepts parameter profanity_filter - [#811](https://github.com/livekit/agents/pull/811) ([@jebjebs](https://github.com/jebjebs))

## 0.6.7

### Patch Changes

- Only send actual audio to Deepgram using a basic audio RMS filter - [#738](https://github.com/livekit/agents/pull/738) ([@keepingitneil](https://github.com/keepingitneil))

- defaults to nova-2-general model - [#726](https://github.com/livekit/agents/pull/726) ([@davidzhao](https://github.com/davidzhao))

## 0.6.6

### Patch Changes

- deepgram: switch the default model to phonecall - [#676](https://github.com/livekit/agents/pull/676) ([@theomonnom](https://github.com/theomonnom))

## 0.6.5

### Patch Changes

- deepgram: fallback to nova-2-general when the language isn't supported - [#623](https://github.com/livekit/agents/pull/623) ([@theomonnom](https://github.com/theomonnom))

## 0.6.4

### Patch Changes

- deepgram: add support for keywords boost/penalty - [#599](https://github.com/livekit/agents/pull/599) ([@theomonnom](https://github.com/theomonnom))

- fix log warnings & cartesia end of speech - [#603](https://github.com/livekit/agents/pull/603) ([@theomonnom](https://github.com/theomonnom))

- stt/tts: fix unread inputs when the input channel is closed - [#594](https://github.com/livekit/agents/pull/594) ([@theomonnom](https://github.com/theomonnom))

## 0.6.3

### Patch Changes

- deepgram: update default model to nova-2-conversationalai - [#576](https://github.com/livekit/agents/pull/576) ([@theomonnom](https://github.com/theomonnom))

## 0.6.2

### Patch Changes

- deepgram: reduce chunks size to 100ms - [#561](https://github.com/livekit/agents/pull/561) ([@theomonnom](https://github.com/theomonnom))

- deepgram: segment audio frames into 200ms intervals before sending to the websocket #549 - [#553](https://github.com/livekit/agents/pull/553) ([@theomonnom](https://github.com/theomonnom))

## 0.6.1

### Patch Changes

- fix end_input not flushing & unhandled flush messages - [#528](https://github.com/livekit/agents/pull/528) ([@theomonnom](https://github.com/theomonnom))

## 0.6.0

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

## 0.6.0-dev.7

### Patch Changes

- pull: '--rebase --autostash ...' - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.6.0-dev.6

### Patch Changes

- test release - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.6.0-dev.5

### Patch Changes

- test release - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.6.0-dev.4

### Patch Changes

- fix changesets release CI - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.6.0-dev.3

### Patch Changes

- bump versions to update dependencies - [#510](https://github.com/livekit/agents/pull/510) ([@theomonnom](https://github.com/theomonnom))

## 0.6.0-dev.2

### Patch Changes

- dev fixes - multiprocessing & voiceassistant - [#493](https://github.com/livekit/agents/pull/493) ([@theomonnom](https://github.com/theomonnom))

## 0.6.0-dev.1

### Minor Changes

- dev prerelease - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.5.2-dev.0

### Patch Changes

- Default loglevel to warn - [#472](https://github.com/livekit/agents/pull/472) ([@lukasIO](https://github.com/lukasIO))
