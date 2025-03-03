# livekit-plugins-anthropic

## 0.2.12

### Patch Changes

- don't pass functions in params when tool choice is set to none - [#1552](https://github.com/livekit/agents/pull/1552) ([@jayeshp19](https://github.com/jayeshp19))

## 0.2.11

### Patch Changes

- Add cache support for Anthropic - [#1478](https://github.com/livekit/agents/pull/1478) ([@jayeshp19](https://github.com/jayeshp19))

## 0.2.10

### Patch Changes

- Added an additional field in LLM capabilities class to check if model providers support function call history within chat context without needing function definitions. - [#1441](https://github.com/livekit/agents/pull/1441) ([@jayeshp19](https://github.com/jayeshp19))

## 0.2.9

### Patch Changes

- improved handling of LLM errors, do not retry if already began - [#1298](https://github.com/livekit/agents/pull/1298) ([@davidzhao](https://github.com/davidzhao))

## 0.2.8

### Patch Changes

- Moved create_ai_function_info to function_context.py for better reusability and reduce repetation - [#1260](https://github.com/livekit/agents/pull/1260) ([@jayeshp19](https://github.com/jayeshp19))

- Add support for OpenAI's "detail" parameter to ChatImage - [#1213](https://github.com/livekit/agents/pull/1213) ([@bcherry](https://github.com/bcherry))

  Add support for data URLs on ChatImage in the Anthropic plugin.

- fix: correctly parse function argument types - [#1221](https://github.com/livekit/agents/pull/1221) ([@jayeshp19](https://github.com/jayeshp19))

- Fix center_aspect_fit bug, add scale_aspect_fit and scale_aspect_fill resizing options. - [#1222](https://github.com/livekit/agents/pull/1222) ([@bcherry](https://github.com/bcherry))

  Make scale_aspect_fit the new default resizing option for video frames.

## 0.2.7

### Patch Changes

- fix: return structured output from func calls - [#1187](https://github.com/livekit/agents/pull/1187) ([@jayeshp19](https://github.com/jayeshp19))

## 0.2.6

### Patch Changes

- fix release - [#1176](https://github.com/livekit/agents/pull/1176) ([@theomonnom](https://github.com/theomonnom))

## 0.2.5

### Patch Changes

- support for custom tool use in LLMs - [#1102](https://github.com/livekit/agents/pull/1102) ([@jayeshp19](https://github.com/jayeshp19))

- feat: llm retry & llm.FallbackAdapter - [#1132](https://github.com/livekit/agents/pull/1132) ([@theomonnom](https://github.com/theomonnom))

## 0.2.4

### Patch Changes

- anthropic tool fix - [#1051](https://github.com/livekit/agents/pull/1051) ([@jayeshp19](https://github.com/jayeshp19))

## 0.2.3

### Patch Changes

- fix: invalid request on anthropic - [#1018](https://github.com/livekit/agents/pull/1018) ([@theomonnom](https://github.com/theomonnom))

## 0.2.2

### Patch Changes

- pipelineagent: expose timing metrics & api errors wip - [#957](https://github.com/livekit/agents/pull/957) ([@theomonnom](https://github.com/theomonnom))

## 0.2.1

### Patch Changes

- Fixes to Anthropic Function Calling - [#708](https://github.com/livekit/agents/pull/708) ([@keepingitneil](https://github.com/keepingitneil))

## 0.2.0

### Minor Changes

- bump anthropic for release - [#724](https://github.com/livekit/agents/pull/724) ([@theomonnom](https://github.com/theomonnom))
