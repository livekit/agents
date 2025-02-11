# livekit-plugins-eou

## 0.4.1

### Patch Changes

- fix incorrect dtype on windows - [#1452](https://github.com/livekit/agents/pull/1452) ([@jeradf](https://github.com/jeradf))

- adjust default probability cutoff - [#1465](https://github.com/livekit/agents/pull/1465) ([@jeradf](https://github.com/jeradf))

## 0.4.0

### Minor Changes

- more accurate, smaller, faster model - [#1426](https://github.com/livekit/agents/pull/1426) ([@jeradf](https://github.com/jeradf))

## 0.3.6

### Patch Changes

- prevent arbitrarily long inputs being passed to turn detector - [#1345](https://github.com/livekit/agents/pull/1345) ([@jeradf](https://github.com/jeradf))

- add timeout for EOU inference requests made to the inference process - [#1315](https://github.com/livekit/agents/pull/1315) ([@theomonnom](https://github.com/theomonnom))

## 0.3.5

### Patch Changes

- fix int32/64 errors on Windows - [#1285](https://github.com/livekit/agents/pull/1285) ([@nbsp](https://github.com/nbsp))

## 0.3.4

### Patch Changes

- add jinja2 dependency to turn detector - [#1277](https://github.com/livekit/agents/pull/1277) ([@davidzhao](https://github.com/davidzhao))

## 0.3.3

### Patch Changes

- use quantized onnx version of turn detector model - [#1231](https://github.com/livekit/agents/pull/1231) ([@jeradf](https://github.com/jeradf))

- use onnxruntime for turn detection and remove pytorch dependency - [#1257](https://github.com/livekit/agents/pull/1257) ([@jeradf](https://github.com/jeradf))

## 0.3.2

### Patch Changes

- improvements to endpointing latency - [#1212](https://github.com/livekit/agents/pull/1212) ([@davidzhao](https://github.com/davidzhao))

- Improvements to end of turn plugin, ensure STT language settings. - [#1195](https://github.com/livekit/agents/pull/1195) ([@davidzhao](https://github.com/davidzhao))

## 0.3.1

### Patch Changes

- fix release - [#1176](https://github.com/livekit/agents/pull/1176) ([@theomonnom](https://github.com/theomonnom))

## 0.3.0

### Minor Changes

- feat: inference process & end of utterance plugin - [#1133](https://github.com/livekit/agents/pull/1133) ([@theomonnom](https://github.com/theomonnom))
