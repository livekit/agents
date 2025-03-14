# livekit-plugins-google

## 0.11.1

### Patch Changes

- allow configurable api version in gemini realtime - [#1656](https://github.com/livekit/agents/pull/1656) ([@jayeshp19](https://github.com/jayeshp19))

## 0.11.0

### Minor Changes

- Add simple video input support for gemini live - [#1536](https://github.com/livekit/agents/pull/1536) ([@bcherry](https://github.com/bcherry))

### Patch Changes

- use streaming AudioDecoder to handle compressed encoding - [#1584](https://github.com/livekit/agents/pull/1584) ([@davidzhao](https://github.com/davidzhao))

- updated livekit-agent reference to <1.0 - [#1607](https://github.com/livekit/agents/pull/1607) ([@davidzhao](https://github.com/davidzhao))

## 0.10.6

### Patch Changes

- google stt: change default model to `latest_long` - [#1552](https://github.com/livekit/agents/pull/1552) ([@jayeshp19](https://github.com/jayeshp19))

- feat: connection pooling. speeds up generation with STT/TTS providers - [#1538](https://github.com/livekit/agents/pull/1538) ([@davidzhao](https://github.com/davidzhao))

- fix: functioncall cancellation ids in realtime - [#1572](https://github.com/livekit/agents/pull/1572) ([@jayeshp19](https://github.com/jayeshp19))

- google-genai version bump & remove id feild from function call and function response - [#1559](https://github.com/livekit/agents/pull/1559) ([@jayeshp19](https://github.com/jayeshp19))

## 0.10.5

### Patch Changes

- fix(google): require min confidence score due to aggressive generation - [#1507](https://github.com/livekit/agents/pull/1507) ([@davidzhao](https://github.com/davidzhao))

## 0.10.4

### Patch Changes

- Gemini realtime : rollback default model to `gemini-2.0-flash-exp` - [#1489](https://github.com/livekit/agents/pull/1489) ([@jayeshp19](https://github.com/jayeshp19))

## 0.10.3

### Patch Changes

- Gemini Realtime: Transcribe model audio via gemini api & use latest model as default for google plugin - [#1446](https://github.com/livekit/agents/pull/1446) ([@jayeshp19](https://github.com/jayeshp19))

- Update to support passing chirp_2 location for other STT credentials - [#1098](https://github.com/livekit/agents/pull/1098) ([@brightsparc](https://github.com/brightsparc))

- Added an additional field in LLM capabilities class to check if model providers support function call history within chat context without needing function definitions. - [#1441](https://github.com/livekit/agents/pull/1441) ([@jayeshp19](https://github.com/jayeshp19))

## 0.10.2

### Patch Changes

- gemini-realtime: fix input audio sample rate - [#1411](https://github.com/livekit/agents/pull/1411) ([@jayeshp19](https://github.com/jayeshp19))

- chore: Replace ValueError with logger.warning for missing GOOGLE_APPLICATION_CREDENTIALS environment variable - [#1415](https://github.com/livekit/agents/pull/1415) ([@hironow](https://github.com/hironow))

## 0.10.1

### Patch Changes

- fix: update default model to chirp2 in google stt & update generate_reply method in gemini realtime - [#1401](https://github.com/livekit/agents/pull/1401) ([@jayeshp19](https://github.com/jayeshp19))

## 0.10.0

### Minor Changes

- support gemini LLM - [#1382](https://github.com/livekit/agents/pull/1382) ([@jayeshp19](https://github.com/jayeshp19))

### Patch Changes

- fix: address breaking change from google-genai >= 0.3.0 - [#1383](https://github.com/livekit/agents/pull/1383) ([@jayeshp19](https://github.com/jayeshp19))

- gemini improvements: exception handling, transcription & Ensure contents.parts is non-empty in gemini contex - [#1398](https://github.com/livekit/agents/pull/1398) ([@jayeshp19](https://github.com/jayeshp19))

- support transcriber session for user/agent audio - [#1321](https://github.com/livekit/agents/pull/1321) ([@jayeshp19](https://github.com/jayeshp19))

## 0.9.1

### Patch Changes

- fetch fresh client on update location and small fix for max_session_duration (4 mins) - [#1342](https://github.com/livekit/agents/pull/1342) ([@jayeshp19](https://github.com/jayeshp19))

- fix Google STT handling of session timeouts - [#1337](https://github.com/livekit/agents/pull/1337) ([@davidzhao](https://github.com/davidzhao))

## 0.9.0

### Minor Changes

- make multimodal class generic and support gemini live api - [#1240](https://github.com/livekit/agents/pull/1240) ([@jayeshp19](https://github.com/jayeshp19))

### Patch Changes

- fix: Ensure STT exceptions are being propagated - [#1291](https://github.com/livekit/agents/pull/1291) ([@davidzhao](https://github.com/davidzhao))

## 0.8.1

### Patch Changes

- fix release - [#1176](https://github.com/livekit/agents/pull/1176) ([@theomonnom](https://github.com/theomonnom))

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
