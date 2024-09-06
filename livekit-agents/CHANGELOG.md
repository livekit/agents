# livekit-agents

## 0.8.10

### Patch Changes

- Pass JobContext to participant entrypoint function - [#694](https://github.com/livekit/agents/pull/694) ([@davidzhao](https://github.com/davidzhao))

- voiceassistant: keep punctuations when sending agent transcription - [#648](https://github.com/livekit/agents/pull/648) ([@theomonnom](https://github.com/theomonnom))

## 0.8.9

### Patch Changes

- Introduce easy api for starting tasks for remote participants - [#679](https://github.com/livekit/agents/pull/679) ([@keepingitneil](https://github.com/keepingitneil))

- update livekit to 0.14.0 and await tracksubscribed - [#678](https://github.com/livekit/agents/pull/678) ([@nbsp](https://github.com/nbsp))

## 0.8.8

### Patch Changes

- fix uninitialized SpeechHandle error on interruption - [#665](https://github.com/livekit/agents/pull/665) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: avoid stacking assistant replies when allow_interruptions=False - [#667](https://github.com/livekit/agents/pull/667) ([@theomonnom](https://github.com/theomonnom))

- fix: disconnect event may now have a arguments - [#668](https://github.com/livekit/agents/pull/668) ([@theomonnom](https://github.com/theomonnom))

- Add ServerMessage.termination handler - [#635](https://github.com/livekit/agents/pull/635) ([@nbsp](https://github.com/nbsp))

## 0.8.7

### Patch Changes

- voiceassistant: fix llm not having the full chat context on bad interruption timing - [#659](https://github.com/livekit/agents/pull/659) ([@theomonnom](https://github.com/theomonnom))

## 0.8.6

### Patch Changes

- voiceassistant: fix will_synthesize_assistant_reply race - [#638](https://github.com/livekit/agents/pull/638) ([@theomonnom](https://github.com/theomonnom))

- Switch Cartesia to a sentence tokenizer and keep the same context id throughout. - [#608](https://github.com/livekit/agents/pull/608) ([@keepingitneil](https://github.com/keepingitneil))
  Propagate segment_id through the basic sentence tokenizer

- silero: adjust vad activation threshold - [#639](https://github.com/livekit/agents/pull/639) ([@theomonnom](https://github.com/theomonnom))

- limit simultaneous process initialization - [#621](https://github.com/livekit/agents/pull/621) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: remove fade effect when interrupting #622 - [#623](https://github.com/livekit/agents/pull/623) ([@theomonnom](https://github.com/theomonnom))

- ipc improvements, fix slow shutdown & cleanup leaked resources - [#607](https://github.com/livekit/agents/pull/607) ([@theomonnom](https://github.com/theomonnom))

- ipc: use our own duplex instead of mp.Queue - [#634](https://github.com/livekit/agents/pull/634) ([@theomonnom](https://github.com/theomonnom))

- Support OpenAI Assistants API as a beta feature under `livekit.plugins.openai.beta` - [#601](https://github.com/livekit/agents/pull/601) ([@keepingitneil](https://github.com/keepingitneil))
  Add \_metadata to ChatCtx and ChatMessage which can be used (in the case of OpenAI assistants) for bookeeping to sync local state with remote, OpenAI state

- llm: fix optional arguments & non-hashable list - [#637](https://github.com/livekit/agents/pull/637) ([@theomonnom](https://github.com/theomonnom))

- silero: fix vad padding & static audio - [#631](https://github.com/livekit/agents/pull/631) ([@theomonnom](https://github.com/theomonnom))

## 0.8.5

### Patch Changes

- add support for optional arguments on ai_callable functions - [#600](https://github.com/livekit/agents/pull/600) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: correctly export AssistantTranscriptionOptions - [#598](https://github.com/livekit/agents/pull/598) ([@theomonnom](https://github.com/theomonnom))

- fix: log levelname not present when using the start subcommand - [#602](https://github.com/livekit/agents/pull/602) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: fix incomplete committed agent transcript in the chat_ctx - [#595](https://github.com/livekit/agents/pull/595) ([@theomonnom](https://github.com/theomonnom))

- cartesia: correctly add spaces & fix tests - [#591](https://github.com/livekit/agents/pull/591) ([@theomonnom](https://github.com/theomonnom))

## 0.8.4

### Patch Changes

- voiceassistant: only commit the spoken words in the chat context. - [#589](https://github.com/livekit/agents/pull/589) ([@theomonnom](https://github.com/theomonnom))

- use aiodns by default - [#579](https://github.com/livekit/agents/pull/579) ([@theomonnom](https://github.com/theomonnom))

- voice_assistant: fix missing spaces between transcript chunks - [#566](https://github.com/livekit/agents/pull/566) ([@egoldschmidt](https://github.com/egoldschmidt))

- voiceassistant: fix transcription being fully sent even when interrupted - [#581](https://github.com/livekit/agents/pull/581) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: fix AssertionError when there is no user_question - [#582](https://github.com/livekit/agents/pull/582) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: fix speech validation cancellation - [#584](https://github.com/livekit/agents/pull/584) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: fix synthesis continuing after interruption - [#588](https://github.com/livekit/agents/pull/588) ([@theomonnom](https://github.com/theomonnom))

## 0.8.3

### Patch Changes

- voiceassistant: run function calls sequentially - [#554](https://github.com/livekit/agents/pull/554) ([@theomonnom](https://github.com/theomonnom))

- configure plugins loggers & more debug logs on the voiceassistant - [#555](https://github.com/livekit/agents/pull/555) ([@theomonnom](https://github.com/theomonnom))

- warn no room connection after job_entry was called after 10 seconds. - [#558](https://github.com/livekit/agents/pull/558) ([@theomonnom](https://github.com/theomonnom))

- deepgram: reduce chunks size to 100ms - [#561](https://github.com/livekit/agents/pull/561) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: cleanup validation behaviour #545 - [#553](https://github.com/livekit/agents/pull/553) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: commit user question directly when allow_interruptions=False - [#547](https://github.com/livekit/agents/pull/547) ([@theomonnom](https://github.com/theomonnom))

- ipc: increase high ping threshold - [#556](https://github.com/livekit/agents/pull/556) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: interrupt on final transcript - [#546](https://github.com/livekit/agents/pull/546) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: tweaks & fix speech being removed too soon from the queue - [#560](https://github.com/livekit/agents/pull/560) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: fix duplicate answers - [#548](https://github.com/livekit/agents/pull/548) ([@theomonnom](https://github.com/theomonnom))

- reduce the default load threshold to a more appropriate default - [#559](https://github.com/livekit/agents/pull/559) ([@theomonnom](https://github.com/theomonnom))

## 0.8.2

### Patch Changes

- fix: remove unnecessary async function - [#540](https://github.com/livekit/agents/pull/540) ([@Nabil372](https://github.com/Nabil372))

## 0.8.1

### Patch Changes

- update livekit-rtc to v0.12.0 - [#535](https://github.com/livekit/agents/pull/535) ([@theomonnom](https://github.com/theomonnom))

- automatically create stt.StreamAdapter when provided stt doesn't support streaming - [#536](https://github.com/livekit/agents/pull/536) ([@theomonnom](https://github.com/theomonnom))

- update examples to the latest API & export AutoSubscribe - [#534](https://github.com/livekit/agents/pull/534) ([@theomonnom](https://github.com/theomonnom))

- fix end_input not flushing & unhandled flush messages - [#528](https://github.com/livekit/agents/pull/528) ([@theomonnom](https://github.com/theomonnom))

## 0.8.0

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

## 0.8.0-dev.8

### Patch Changes

- pull: '--rebase --autostash ...' - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.8.0-dev.7

### Patch Changes

- test release - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.8.0-dev.6

### Patch Changes

- test release - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.8.0-dev.5

### Patch Changes

- fix changesets release CI - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.8.0-dev.4

### Patch Changes

- bump versions to update dependencies - [#510](https://github.com/livekit/agents/pull/510) ([@theomonnom](https://github.com/theomonnom))

## 0.8.0-dev.3

### Patch Changes

- dev fixes - multiprocessing & voiceassistant - [#493](https://github.com/livekit/agents/pull/493) ([@theomonnom](https://github.com/theomonnom))

## 0.8.0-dev.2

### Minor Changes

- dev prerelease - [#435](https://github.com/livekit/agents/pull/435) ([@theomonnom](https://github.com/theomonnom))

## 0.7.3-dev.1

### Patch Changes

- Default loglevel to warn - [#472](https://github.com/livekit/agents/pull/472) ([@lukasIO](https://github.com/lukasIO))
