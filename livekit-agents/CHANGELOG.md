# livekit-agents

## 0.12.13

### Patch Changes

- Allow shutdown callbacks to take reason - [#1475](https://github.com/livekit/agents/pull/1475) ([@martin-purplefish](https://github.com/martin-purplefish))

## 0.12.12

### Patch Changes

- fix agent transcription could not be disabled - [#1448](https://github.com/livekit/agents/pull/1448) ([@davidzhao](https://github.com/davidzhao))

- Added an additional field in LLM capabilities class to check if model providers support function call history within chat context without needing function definitions. - [#1441](https://github.com/livekit/agents/pull/1441) ([@jayeshp19](https://github.com/jayeshp19))

- support agent.say inside the before_llm_cb - [#1460](https://github.com/livekit/agents/pull/1460) ([@longcw](https://github.com/longcw))

## 0.12.11

### Patch Changes

- gemini-realtime: fix input audio sample rate - [#1411](https://github.com/livekit/agents/pull/1411) ([@jayeshp19](https://github.com/jayeshp19))

- fix(pipeline_agent): clear user transcript when before_llm_cb returns false - [#1423](https://github.com/livekit/agents/pull/1423) ([@s-hamdananwar](https://github.com/s-hamdananwar))

- fix: fallbackadapter to correctly handle function calls - [#1429](https://github.com/livekit/agents/pull/1429) ([@davidzhao](https://github.com/davidzhao))

- improved TTFB metrics for streaming TTS - [#1431](https://github.com/livekit/agents/pull/1431) ([@davidzhao](https://github.com/davidzhao))

## 0.12.10

### Patch Changes

- fix false positive interruption tripping up certain LLMs - [#1410](https://github.com/livekit/agents/pull/1410) ([@davidzhao](https://github.com/davidzhao))

- fix: ensure llm.FallbackAdapter executes function calls - [#1409](https://github.com/livekit/agents/pull/1409) ([@davidzhao](https://github.com/davidzhao))

## 0.12.9

### Patch Changes

- add generate_reply api for multimodal agent - [#1359](https://github.com/livekit/agents/pull/1359) ([@longcw](https://github.com/longcw))

- remove aiodns from livekit-agents - [#1368](https://github.com/livekit/agents/pull/1368) ([@theomonnom](https://github.com/theomonnom))

## 0.12.8

### Patch Changes

- Fix not awaiting forward task in TTS forwarder, leading to warnings. - [#1339](https://github.com/livekit/agents/pull/1339) ([@martin-purplefish](https://github.com/martin-purplefish))

- reduces initial delay before model retries - [#1337](https://github.com/livekit/agents/pull/1337) ([@davidzhao](https://github.com/davidzhao))

- fix the function calls without a text response are not added to chat ctx - [#1349](https://github.com/livekit/agents/pull/1349) ([@longcw](https://github.com/longcw))

- add timeout for EOU inference requests made to the inference process - [#1315](https://github.com/livekit/agents/pull/1315) ([@theomonnom](https://github.com/theomonnom))

- support disabling server VAD for OpenAI realtime model - [#1347](https://github.com/livekit/agents/pull/1347) ([@longcw](https://github.com/longcw))

## 0.12.7

### Patch Changes

- ensure job status updates contain the correct status - [#1319](https://github.com/livekit/agents/pull/1319) ([@davidzhao](https://github.com/davidzhao))

## 0.12.6

### Patch Changes

- expose worker_id in jobcontext - [#1307](https://github.com/livekit/agents/pull/1307) ([@s-hamdananwar](https://github.com/s-hamdananwar))

- improved handling of LLM errors, do not retry if already began - [#1298](https://github.com/livekit/agents/pull/1298) ([@davidzhao](https://github.com/davidzhao))

- Do not pass function context if at max depth - [#1306](https://github.com/livekit/agents/pull/1306) ([@martin-purplefish](https://github.com/martin-purplefish))

- avoid warnings when function depth matches limit - [#1316](https://github.com/livekit/agents/pull/1316) ([@davidzhao](https://github.com/davidzhao))

- improve interruption handling, avoid agent from getting stuck - [#1290](https://github.com/livekit/agents/pull/1290) ([@davidzhao](https://github.com/davidzhao))

- add manual interrupt method for pipeline agent - [#1294](https://github.com/livekit/agents/pull/1294) ([@longcw](https://github.com/longcw))

- make multimodal class generic and support gemini live api - [#1240](https://github.com/livekit/agents/pull/1240) ([@jayeshp19](https://github.com/jayeshp19))

## 0.12.5

### Patch Changes

- make max_endpoint_delay configurable - [#1277](https://github.com/livekit/agents/pull/1277) ([@davidzhao](https://github.com/davidzhao))

- set USE_DOCSTRING as default for ai_callable - [#1266](https://github.com/livekit/agents/pull/1266) ([@longcw](https://github.com/longcw))

- fix: do not log process warning when process not found - [#1281](https://github.com/livekit/agents/pull/1281) ([@davidzhao](https://github.com/davidzhao))

- fix context when functions have been called - [#1279](https://github.com/livekit/agents/pull/1279) ([@jmugicagonz](https://github.com/jmugicagonz))

## 0.12.4

### Patch Changes

- avoid duplicated chat ctx for function calls with messages - [#1254](https://github.com/livekit/agents/pull/1254) ([@longcw](https://github.com/longcw))

## 0.12.3

### Patch Changes

- Moved create_ai_function_info to function_context.py for better reusability and reduce repetation - [#1260](https://github.com/livekit/agents/pull/1260) ([@jayeshp19](https://github.com/jayeshp19))

- added streaming audio decoder for compressed audio. - [#1236](https://github.com/livekit/agents/pull/1236) ([@davidzhao](https://github.com/davidzhao))

- Add JPEG quality param to image encoder - [#1249](https://github.com/livekit/agents/pull/1249) ([@bcherry](https://github.com/bcherry))

- Add support for OpenAI's "detail" parameter to ChatImage - [#1213](https://github.com/livekit/agents/pull/1213) ([@bcherry](https://github.com/bcherry))

  Add support for data URLs on ChatImage in the Anthropic plugin.

- fix: correctly parse function argument types - [#1221](https://github.com/livekit/agents/pull/1221) ([@jayeshp19](https://github.com/jayeshp19))

- Fix center_aspect_fit bug, add scale_aspect_fit and scale_aspect_fill resizing options. - [#1222](https://github.com/livekit/agents/pull/1222) ([@bcherry](https://github.com/bcherry))

  Make scale_aspect_fit the new default resizing option for video frames.

## 0.12.2

### Patch Changes

- improvements to endpointing latency - [#1212](https://github.com/livekit/agents/pull/1212) ([@davidzhao](https://github.com/davidzhao))

- Improvements to end of turn plugin, ensure STT language settings. - [#1195](https://github.com/livekit/agents/pull/1195) ([@davidzhao](https://github.com/davidzhao))

- fix duplicated agent speech commit for message with function call - [#1192](https://github.com/livekit/agents/pull/1192) ([@longcw](https://github.com/longcw))

- fix: Handle optional func args in tool calls when set to `None` - [#1211](https://github.com/livekit/agents/pull/1211) ([@jayeshp19](https://github.com/jayeshp19))

## 0.12.1

### Patch Changes

- fix release - [#1176](https://github.com/livekit/agents/pull/1176) ([@theomonnom](https://github.com/theomonnom))

## 0.12.0

### Minor Changes

- add nested speech handles, now agent.say works during a function call - [#1130](https://github.com/livekit/agents/pull/1130) ([@longcw](https://github.com/longcw))

### Patch Changes

- feat: stt retry & stt.FallbackAdapter - [#1114](https://github.com/livekit/agents/pull/1114) ([@theomonnom](https://github.com/theomonnom))

- expose LiveKitAPI from the a JobContext - [#1159](https://github.com/livekit/agents/pull/1159) ([@theomonnom](https://github.com/theomonnom))

- add extra chat messages to the end of the function call outputs - [#1165](https://github.com/livekit/agents/pull/1165) ([@longcw](https://github.com/longcw))

- Add retries to recover from text mode to audio model for realtime API - [#1121](https://github.com/livekit/agents/pull/1121) ([@longcw](https://github.com/longcw))

- prepare for release - [#1160](https://github.com/livekit/agents/pull/1160) ([@theomonnom](https://github.com/theomonnom))

- add max_job_memory_usage and will kill the job if it exceeds the limit - [#1136](https://github.com/livekit/agents/pull/1136) ([@longcw](https://github.com/longcw))

- support for custom tool use in LLMs - [#1102](https://github.com/livekit/agents/pull/1102) ([@jayeshp19](https://github.com/jayeshp19))

- feat: tts retry & tts.FallbackAdapter - [#1074](https://github.com/livekit/agents/pull/1074) ([@theomonnom](https://github.com/theomonnom))

- Expose multimodal agent metrics - [#1080](https://github.com/livekit/agents/pull/1080) ([@longcw](https://github.com/longcw))

- preload mp3 decoder for TTS plugins - [#1129](https://github.com/livekit/agents/pull/1129) ([@jayeshp19](https://github.com/jayeshp19))

- feat: llm retry & llm.FallbackAdapter - [#1132](https://github.com/livekit/agents/pull/1132) ([@theomonnom](https://github.com/theomonnom))

- feat: inference process & end of utterance plugin - [#1133](https://github.com/livekit/agents/pull/1133) ([@theomonnom](https://github.com/theomonnom))

- vertex ai support with openai library - [#1084](https://github.com/livekit/agents/pull/1084) ([@jayeshp19](https://github.com/jayeshp19))

## 0.11.3

### Patch Changes

- add PeriodicCollector utility for metrics - [#1094](https://github.com/livekit/agents/pull/1094) ([@davidzhao](https://github.com/davidzhao))

## 0.11.2

### Patch Changes

- Fix interrupt_min_words handling - [#1062](https://github.com/livekit/agents/pull/1062) ([@davidzhao](https://github.com/davidzhao))

- pipelineagent: fix speech_committed never called - [#1078](https://github.com/livekit/agents/pull/1078) ([@theomonnom](https://github.com/theomonnom))

- Allow setting agent attributes when accepting job - [#1076](https://github.com/livekit/agents/pull/1076) ([@davidzhao](https://github.com/davidzhao))

- handles error in function calls - [#1057](https://github.com/livekit/agents/pull/1057) ([@jayeshp19](https://github.com/jayeshp19))

- Include job count in WorkerStatus and pass in worker for load_fnc - [#1046](https://github.com/livekit/agents/pull/1046) ([@keepingitneil](https://github.com/keepingitneil))

- Fix delay calculation - [#1081](https://github.com/livekit/agents/pull/1081) ([@martin-purplefish](https://github.com/martin-purplefish))

- sync the Realtime API converstation items and add set_chat_ctx - [#1015](https://github.com/livekit/agents/pull/1015) ([@longcw](https://github.com/longcw))

- added metrics for idle time - [#1064](https://github.com/livekit/agents/pull/1064) ([@jayeshp19](https://github.com/jayeshp19))

## 0.11.1

### Patch Changes

- Fix stack dump on closed stream - [#1023](https://github.com/livekit/agents/pull/1023) ([@martin-purplefish](https://github.com/martin-purplefish))

- fix: invalid request on anthropic - [#1018](https://github.com/livekit/agents/pull/1018) ([@theomonnom](https://github.com/theomonnom))

- fix: IndexError on tts metrics - [#1028](https://github.com/livekit/agents/pull/1028) ([@theomonnom](https://github.com/theomonnom))

## 0.11.0

### Minor Changes

- prepare for release - [#1007](https://github.com/livekit/agents/pull/1007) ([@theomonnom](https://github.com/theomonnom))

### Patch Changes

- Fix race in load calc initialization - [#969](https://github.com/livekit/agents/pull/969) ([@martin-purplefish](https://github.com/martin-purplefish))

- Fix incorrect load computation on docker instances - [#972](https://github.com/livekit/agents/pull/972) ([@martin-purplefish](https://github.com/martin-purplefish))

- stt: reduce bandwidth usage by reducing sample_rate to 16khz - [#920](https://github.com/livekit/agents/pull/920) ([@theomonnom](https://github.com/theomonnom))

- Reorganized metrics, added create_metrics_logger - [#1009](https://github.com/livekit/agents/pull/1009) ([@davidzhao](https://github.com/davidzhao))

- pipelineagent: expose timing metrics & api errors wip - [#957](https://github.com/livekit/agents/pull/957) ([@theomonnom](https://github.com/theomonnom))

- Allow kind to be list or single value - [#1006](https://github.com/livekit/agents/pull/1006) ([@keepingitneil](https://github.com/keepingitneil))

- fix before_llm_cb not handling coroutines returning False - [#961](https://github.com/livekit/agents/pull/961) ([@Tanesan](https://github.com/Tanesan))

- expose transcriptions for multimodal agents - [#1001](https://github.com/livekit/agents/pull/1001) ([@longcw](https://github.com/longcw))

- Fix stack dump on room shutdown - [#989](https://github.com/livekit/agents/pull/989) ([@martin-purplefish](https://github.com/martin-purplefish))

- Add exception logging for tool calls - [#923](https://github.com/livekit/agents/pull/923) ([@martin-purplefish](https://github.com/martin-purplefish))

- Skip egress by default in participant-related utilities on JobContext - [#1005](https://github.com/livekit/agents/pull/1005) ([@keepingitneil](https://github.com/keepingitneil))

- pipeline-agent: avoid nested function calls - [#935](https://github.com/livekit/agents/pull/935) ([@theomonnom](https://github.com/theomonnom))

- expose usage metrics - [#984](https://github.com/livekit/agents/pull/984) ([@theomonnom](https://github.com/theomonnom))

- fix jobs never reloading - [#934](https://github.com/livekit/agents/pull/934) ([@theomonnom](https://github.com/theomonnom))

- voicepipeline: support recursive/chained function calls - [#970](https://github.com/livekit/agents/pull/970) ([@theomonnom](https://github.com/theomonnom))

## 0.10.2

### Patch Changes

- Fix split_paragraphs and simple-rag example - [#896](https://github.com/livekit/agents/pull/896) ([@davidzhao](https://github.com/davidzhao))

- Fix bug where if the tts_source was a string but before_tts_cb returned AsyncIterable[str], the transcript would not be synthesized. - [#906](https://github.com/livekit/agents/pull/906) ([@martin-purplefish](https://github.com/martin-purplefish))

- Allow forcing interruptions of incomplete audio - [#891](https://github.com/livekit/agents/pull/891) ([@martin-purplefish](https://github.com/martin-purplefish))

- Include chat context on collected tool calls - [#897](https://github.com/livekit/agents/pull/897) ([@martin-purplefish](https://github.com/martin-purplefish))

## 0.10.1

### Patch Changes

- use rtc.combine_audio_frames - [#841](https://github.com/livekit/agents/pull/841) ([@theomonnom](https://github.com/theomonnom))

- Fix agent state to not change to listening when user speaks - [#857](https://github.com/livekit/agents/pull/857) ([@martin-purplefish](https://github.com/martin-purplefish))
  Fixed canceling uncancelable speech
  Fixed bug where agent would get stuck with uninterruptable speech.

- Fix bug where empty audio would cause agent to get stuck. - [#836](https://github.com/livekit/agents/pull/836) ([@martin-purplefish](https://github.com/martin-purplefish))

- fix: handle when STT does not return any speech - [#854](https://github.com/livekit/agents/pull/854) ([@davidzhao](https://github.com/davidzhao))

- Fix watcher reloaded processes double connecting to rooms - [#822](https://github.com/livekit/agents/pull/822) ([@keepingitneil](https://github.com/keepingitneil))

- voice-pipeline: avoid stacked replies when interruptions is disallowed - [#869](https://github.com/livekit/agents/pull/869) ([@theomonnom](https://github.com/theomonnom))

- disable preemptive_synthesis by default - [#867](https://github.com/livekit/agents/pull/867) ([@theomonnom](https://github.com/theomonnom))

- Fixed bug where agent would get stuck on non-interruptable speech - [#850](https://github.com/livekit/agents/pull/850) ([@martin-purplefish](https://github.com/martin-purplefish))

- use EventEmitter from rtc - [#879](https://github.com/livekit/agents/pull/879) ([@theomonnom](https://github.com/theomonnom))

- AudioByteStream: avoid empty frames on flush - [#840](https://github.com/livekit/agents/pull/840) ([@theomonnom](https://github.com/theomonnom))

- improve worker logs - [#878](https://github.com/livekit/agents/pull/878) ([@theomonnom](https://github.com/theomonnom))

- voice-pipeline: fix tts_forwarder not always being closed - [#871](https://github.com/livekit/agents/pull/871) ([@theomonnom](https://github.com/theomonnom))

- bump livekit-rtc to v0.17.5 - [#880](https://github.com/livekit/agents/pull/880) ([@theomonnom](https://github.com/theomonnom))

- Fixed bug where agent would freeze if before_llm_cb returned false - [#865](https://github.com/livekit/agents/pull/865) ([@martin-purplefish](https://github.com/martin-purplefish))

## 0.10.0

### Minor Changes

- OpenAI Realtime API support - [#814](https://github.com/livekit/agents/pull/814) ([@theomonnom](https://github.com/theomonnom))

### Patch Changes

- bump livekit to v0.17.2 - [#815](https://github.com/livekit/agents/pull/815) ([@theomonnom](https://github.com/theomonnom))

- silero: support any sample rate - [#805](https://github.com/livekit/agents/pull/805) ([@theomonnom](https://github.com/theomonnom))

## 0.9.1

### Patch Changes

- fix VoiceAssisstant being stuck when interrupting before user speech is committed - [#790](https://github.com/livekit/agents/pull/790) ([@coderlxn](https://github.com/coderlxn))

- Fix function for OpenAI Assistants - [#784](https://github.com/livekit/agents/pull/784) ([@keepingitneil](https://github.com/keepingitneil))

## 0.9.0

### Minor Changes

- rename voice_assistant.state to lk.agent.state - [#772](https://github.com/livekit/agents/pull/772) ([@bcherry](https://github.com/bcherry))

### Patch Changes

- bump rtc - [#782](https://github.com/livekit/agents/pull/782) ([@nbsp](https://github.com/nbsp))

- improve graceful shutdown - [#756](https://github.com/livekit/agents/pull/756) ([@theomonnom](https://github.com/theomonnom))

- avoid returning tiny frames from TTS - [#747](https://github.com/livekit/agents/pull/747) ([@theomonnom](https://github.com/theomonnom))

- windows: default to threaded executor & fix dev mode - [#755](https://github.com/livekit/agents/pull/755) ([@theomonnom](https://github.com/theomonnom))

- 11labs: send phoneme in one entire xml chunk - [#766](https://github.com/livekit/agents/pull/766) ([@theomonnom](https://github.com/theomonnom))

- fix: process not starting if num_idle_processes is zero - [#763](https://github.com/livekit/agents/pull/763) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: avoid tiny frames on playout - [#750](https://github.com/livekit/agents/pull/750) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: expose turn_completion_delay - [#752](https://github.com/livekit/agents/pull/752) ([@theomonnom](https://github.com/theomonnom))

- limit concurrent process init to 1 - [#751](https://github.com/livekit/agents/pull/751) ([@theomonnom](https://github.com/theomonnom))

- Add typing-extensions as a dependency - [#778](https://github.com/livekit/agents/pull/778) ([@keepingitneil](https://github.com/keepingitneil))

- Allow setting LLM temperature with VoiceAssistant - [#741](https://github.com/livekit/agents/pull/741) ([@davidzhao](https://github.com/davidzhao))

- better dev defaults - [#762](https://github.com/livekit/agents/pull/762) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: allow to cancel llm generation inside before_llm_cb - [#753](https://github.com/livekit/agents/pull/753) ([@theomonnom](https://github.com/theomonnom))

- use os.exit to exit forcefully - [#770](https://github.com/livekit/agents/pull/770) ([@theomonnom](https://github.com/theomonnom))

## 0.8.12

### Patch Changes

- tts*forwarder: don't raise inside mark*{audio,text}\_segment_end when nothing was pushed - [#730](https://github.com/livekit/agents/pull/730) ([@theomonnom](https://github.com/theomonnom))

## 0.8.11

### Patch Changes

- improve gracefully_cancel logic - [#720](https://github.com/livekit/agents/pull/720) ([@theomonnom](https://github.com/theomonnom))

- Make ctx.room.name available prior to connection - [#716](https://github.com/livekit/agents/pull/716) ([@davidzhao](https://github.com/davidzhao))

- ipc: add threaded job runner - [#684](https://github.com/livekit/agents/pull/684) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: add VoiceAssistantState - [#654](https://github.com/livekit/agents/pull/654) ([@lukasIO](https://github.com/lukasIO))

- add JobContext.wait_for_participant - [#712](https://github.com/livekit/agents/pull/712) ([@theomonnom](https://github.com/theomonnom))

- fix non pickleable log - [#691](https://github.com/livekit/agents/pull/691) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: skip speech initialization if interrupted - [#715](https://github.com/livekit/agents/pull/715) ([@theomonnom](https://github.com/theomonnom))

- bump required livekit version to 0.15.2 - [#722](https://github.com/livekit/agents/pull/722) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: add will_synthesize_assistant_speech - [#706](https://github.com/livekit/agents/pull/706) ([@theomonnom](https://github.com/theomonnom))

- voiceassistant: fix mark_audio_segment_end with no audio data - [#719](https://github.com/livekit/agents/pull/719) ([@theomonnom](https://github.com/theomonnom))

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
