# livekit-plugins-deepgram

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
