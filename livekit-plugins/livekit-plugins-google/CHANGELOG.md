# livekit-plugins-google

## 0.8.0

### Minor Changes

- Add support for google STT chirp_2 model. - [#1089](https://github.com/livekit/agents/pull/1089) ([@brightsparc](https://github.com/brightsparc))

### Patch Changes

- feat: stt retry & stt.FallbackAdapter - [#1114](https://github.com/livekit/agents/pull/1114) ([@theomonnom](https://github.com/theomonnom))

- fix: add retry logic for google stt abort exception - [#1100](https://github.com/livekit/agents/pull/1100) ([@jayeshp19](https://github.com/jayeshp19))

- feat: tts retry & tts.FallbackAdapter - [#1074](https://github.com/livekit/agents/pull/1074) ([@theomonnom](https://github.com/theomonnom))

- google STT - use the baseclass resampler - [#1106](https://github.com/livekit/agents/pull/1106) ([@jayeshp19](https://github.com/jayeshp19))

## 0.7.3

### Patch Changes

- added catch for aborted speech - [#1055](https://github.com/livekit/agents/pull/1055) ([@jayeshp19](https://github.com/jayeshp19))

- Make Google STT keywords match Deepgram - [#1067](https://github.com/livekit/agents/pull/1067) ([@martin-purplefish](https://github.com/martin-purplefish))

- Add support for boosting phrases in Google STT - [#1066](https://github.com/livekit/agents/pull/1066) ([@martin-purplefish](https://github.com/martin-purplefish))

## 0.7.2

### Patch Changes

- add update_options to TTS - [#922](https://github.com/livekit/agents/pull/922) ([@theomonnom](https://github.com/theomonnom))

- Additional options enabled on Google TTS - [#945](https://github.com/livekit/agents/pull/945) ([@hari-truviz](https://github.com/hari-truviz))

- pipelineagent: expose timing metrics & api errors wip - [#957](https://github.com/livekit/agents/pull/957) ([@theomonnom](https://github.com/theomonnom))

- expose usage metrics - [#984](https://github.com/livekit/agents/pull/984) ([@theomonnom](https://github.com/theomonnom))

## 0.7.1

### Patch Changes

- avoid returning tiny frames from TTS - [#747](https://github.com/livekit/agents/pull/747) ([@theomonnom](https://github.com/theomonnom))

## 0.7.0

### Minor Changes

- Enable use of Google STT with Application Default Credentials. - [#721](https://github.com/livekit/agents/pull/721) ([@rsinnet](https://github.com/rsinnet))

### Patch Changes

- google-tts: ignore wav header - [#703](https://github.com/livekit/agents/pull/703) ([@theomonnom](https://github.com/theomonnom))

## 0.6.3

### Patch Changes

- Fix Google STT exception when no valid speech is recognized - [#680](https://github.com/livekit/agents/pull/680) ([@davidzhao](https://github.com/davidzhao))

## 0.6.2

### Patch Changes

- stt/tts: fix unread inputs when the input channel is closed - [#594](https://github.com/livekit/agents/pull/594) ([@theomonnom](https://github.com/theomonnom))

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
