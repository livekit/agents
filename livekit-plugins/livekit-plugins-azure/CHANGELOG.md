# livekit-plugins-azure

## 0.5.4

### Patch Changes

- Add handlers for supported synthesis events for Azure TTS - [#1486](https://github.com/livekit/agents/pull/1486) ([@anishnag](https://github.com/anishnag))

## 0.5.3

### Patch Changes

- azure speech support all different configs - [#1362](https://github.com/livekit/agents/pull/1362) ([@longcw](https://github.com/longcw))

- reduces initial delay before model retries - [#1337](https://github.com/livekit/agents/pull/1337) ([@davidzhao](https://github.com/davidzhao))

## 0.5.2

### Patch Changes

- fix: Ensure STT exceptions are being propagated - [#1291](https://github.com/livekit/agents/pull/1291) ([@davidzhao](https://github.com/davidzhao))

## 0.5.1

### Patch Changes

- fix azure stt language autodetection - [#1246](https://github.com/livekit/agents/pull/1246) ([@davidzhao](https://github.com/davidzhao))

## 0.5.0

### Minor Changes

- Improvements to end of turn plugin, ensure STT language settings. - [#1195](https://github.com/livekit/agents/pull/1195) ([@davidzhao](https://github.com/davidzhao))

## 0.4.4

### Patch Changes

- fix release - [#1176](https://github.com/livekit/agents/pull/1176) ([@theomonnom](https://github.com/theomonnom))

## 0.4.3

### Patch Changes

- feat: stt retry & stt.FallbackAdapter - [#1114](https://github.com/livekit/agents/pull/1114) ([@theomonnom](https://github.com/theomonnom))

- azure: support auth entra token for TTS - [#1134](https://github.com/livekit/agents/pull/1134) ([@nfma](https://github.com/nfma))

- feat: tts retry & tts.FallbackAdapter - [#1074](https://github.com/livekit/agents/pull/1074) ([@theomonnom](https://github.com/theomonnom))

## 0.4.2

### Patch Changes

- add support for azure speech containers - [#1043](https://github.com/livekit/agents/pull/1043) ([@longcw](https://github.com/longcw))

- fix azure sample_rate parameter - [#1072](https://github.com/livekit/agents/pull/1072) ([@theomonnom](https://github.com/theomonnom))

## 0.4.1

### Patch Changes

- add update_options to TTS - [#922](https://github.com/livekit/agents/pull/922) ([@theomonnom](https://github.com/theomonnom))

- pipelineagent: expose timing metrics & api errors wip - [#957](https://github.com/livekit/agents/pull/957) ([@theomonnom](https://github.com/theomonnom))

- azure tts: fix SSML Implementation by Adding <voice> Tag - [#929](https://github.com/livekit/agents/pull/929) ([@samirsalman](https://github.com/samirsalman))

- azure tts: fix Prosody Config Validation - [#918](https://github.com/livekit/agents/pull/918) ([@samirsalman](https://github.com/samirsalman))

- expose usage metrics - [#984](https://github.com/livekit/agents/pull/984) ([@theomonnom](https://github.com/theomonnom))

## 0.4.0

### Minor Changes

- Azure TTS Prosody SSML support #912 - [#914](https://github.com/livekit/agents/pull/914) ([@theomonnom](https://github.com/theomonnom))

## 0.3.2

### Patch Changes

- avoid returning tiny frames from TTS - [#747](https://github.com/livekit/agents/pull/747) ([@theomonnom](https://github.com/theomonnom))

## 0.3.1

### Patch Changes

- fix end_input not flushing & unhandled flush messages - [#528](https://github.com/livekit/agents/pull/528) ([@theomonnom](https://github.com/theomonnom))

## 0.3.0

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

## 0.3.0-dev.7

### Patch Changes

- pull: '--rebase --autostash ...' - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.3.0-dev.6

### Patch Changes

- test release - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.3.0-dev.5

### Patch Changes

- test release - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.3.0-dev.4

### Patch Changes

- fix changesets release CI - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.3.0-dev.3

### Patch Changes

- bump versions to update dependencies - [#510](https://github.com/livekit/agents/pull/510) ([@theomonnom](https://github.com/theomonnom))

## 0.3.0-dev.2

### Patch Changes

- dev fixes - multiprocessing & voiceassistant - [#493](https://github.com/livekit/agents/pull/493) ([@theomonnom](https://github.com/theomonnom))

## 0.3.0-dev.1

### Minor Changes

- dev prerelease - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.2.2-dev.0

### Patch Changes

- Default loglevel to warn - [#472](https://github.com/livekit/agents/pull/472) ([@lukasIO](https://github.com/lukasIO))
