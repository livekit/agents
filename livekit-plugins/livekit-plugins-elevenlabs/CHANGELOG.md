# livekit-plugins-elevenlabs

## 0.8.1

### Patch Changes

- Revert to using 'isFinal' in ElevenLabs for reliable audio packet completion detection - [#1676](https://github.com/livekit/agents/pull/1676) ([@jayeshp19](https://github.com/jayeshp19))

## 0.8.0

### Minor Changes

- use streaming AudioDecoder to handle compressed encoding - [#1584](https://github.com/livekit/agents/pull/1584) ([@davidzhao](https://github.com/davidzhao))

### Patch Changes

- added a tts.prewarm method to start the connection pool early. - [#1587](https://github.com/livekit/agents/pull/1587) ([@davidzhao](https://github.com/davidzhao))

- deprecated elevenlabs' optimize_stream_latency option - [#1587](https://github.com/livekit/agents/pull/1587) ([@davidzhao](https://github.com/davidzhao))

- increase elevenlabs websocket connection timeout to default 300 seconds - [#1582](https://github.com/livekit/agents/pull/1582) ([@jayeshp19](https://github.com/jayeshp19))

- updated livekit-agent reference to <1.0 - [#1607](https://github.com/livekit/agents/pull/1607) ([@davidzhao](https://github.com/davidzhao))

- Added speed parameter for voices. - [#1574](https://github.com/livekit/agents/pull/1574) ([@MatthiasGruba](https://github.com/MatthiasGruba))

  E.g.:

  ```python
  voice = Voice(
      id="EXAVITQu4vr4xnSDxMaL",
      name="Bella",
      category="premade",
      settings=VoiceSettings(
          stability=0.71,
          speed=1.2,
          similarity_boost=0.5,
          style=0.0,
          use_speaker_boost=True,
      ),
  )

  ```

## 0.7.14

### Patch Changes

- use connection pool for elevenlabs websocket persistant connection - [#1546](https://github.com/livekit/agents/pull/1546) ([@jayeshp19](https://github.com/jayeshp19))

- remove update options from tts synthesis stream - [#1546](https://github.com/livekit/agents/pull/1546) ([@jayeshp19](https://github.com/jayeshp19))

## 0.7.13

### Patch Changes

- 11labs: ensure websocket connection is closed properly - [#1468](https://github.com/livekit/agents/pull/1468) ([@davidzhao](https://github.com/davidzhao))

## 0.7.12

### Patch Changes

- improved TTFB metrics for streaming TTS - [#1431](https://github.com/livekit/agents/pull/1431) ([@davidzhao](https://github.com/davidzhao))

## 0.7.11

### Patch Changes

- add latest model by 11labs - [#1396](https://github.com/livekit/agents/pull/1396) ([@jayeshp19](https://github.com/jayeshp19))

## 0.7.10

### Patch Changes

- Add language param to ElevenLabs TTS update_options - [#1333](https://github.com/livekit/agents/pull/1333) ([@cch41](https://github.com/cch41))

## 0.7.9

### Patch Changes

- fix release - [#1176](https://github.com/livekit/agents/pull/1176) ([@theomonnom](https://github.com/theomonnom))

## 0.7.8

### Patch Changes

- feat: tts retry & tts.FallbackAdapter - [#1074](https://github.com/livekit/agents/pull/1074) ([@theomonnom](https://github.com/theomonnom))

## 0.7.7

### Patch Changes

- support language code in ElevenLabs TTS (#985) - [#1029](https://github.com/livekit/agents/pull/1029) ([@theomonnom](https://github.com/theomonnom))

## 0.7.6

### Patch Changes

- add update_options to TTS - [#922](https://github.com/livekit/agents/pull/922) ([@theomonnom](https://github.com/theomonnom))

- pipelineagent: expose timing metrics & api errors wip - [#957](https://github.com/livekit/agents/pull/957) ([@theomonnom](https://github.com/theomonnom))

- expose usage metrics - [#984](https://github.com/livekit/agents/pull/984) ([@theomonnom](https://github.com/theomonnom))

## 0.7.5

### Patch Changes

- avoid returning tiny frames from TTS - [#747](https://github.com/livekit/agents/pull/747) ([@theomonnom](https://github.com/theomonnom))

- 11labs: send phoneme in one entire xml chunk - [#766](https://github.com/livekit/agents/pull/766) ([@theomonnom](https://github.com/theomonnom))

## 0.7.4

### Patch Changes

- elevenlabs: expose enable_ssml_parsing - [#723](https://github.com/livekit/agents/pull/723) ([@theomonnom](https://github.com/theomonnom))

## 0.7.3

### Patch Changes

- elevenlabs: fix send_task not closing properly - [#596](https://github.com/livekit/agents/pull/596) ([@theomonnom](https://github.com/theomonnom))

- Fix elevenlabs voice settings breaking - [#586](https://github.com/livekit/agents/pull/586) ([@nbsp](https://github.com/nbsp))

## 0.7.2

### Patch Changes

- elevenlabs: update default model to eleven_turbo_v2_5 - [#578](https://github.com/livekit/agents/pull/578) ([@theomonnom](https://github.com/theomonnom))

- gracefully error on non-PCM data - [#567](https://github.com/livekit/agents/pull/567) ([@nbsp](https://github.com/nbsp))

## 0.7.1

### Patch Changes

- fix end_input not flushing & unhandled flush messages - [#528](https://github.com/livekit/agents/pull/528) ([@theomonnom](https://github.com/theomonnom))

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
