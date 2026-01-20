# Intelligent Interruption Handler

## What It Does

When the AI agent is talking and you say "yeah" or "ok" to show you're listening, the agent **keeps talking** instead of stopping.

But if you say "stop" or "wait", the agent **stops immediately**.

---

## Quick Summary

| You Say | Agent is... | What Happens |
|---------|-------------|--------------|
| "yeah", "ok", "hmm" | Talking | Agent **continues** |
| "stop", "wait", "no" | Talking | Agent **stops** |
| "yeah", "ok", "hmm" | Silent | Agent **responds** |
| "yeah wait a second" | Talking | Agent **stops** (has "wait") |

---

## How to Run

```bash
# 1. Install
cd livekit-agents
pip install -e .

# 2. Set your keys
export LIVEKIT_URL="wss://your-server"
export LIVEKIT_API_KEY="your-key"
export LIVEKIT_API_SECRET="your-secret"
export OPENAI_API_KEY="your-openai-key"

# 3. Run
python examples/voice_agents/minimal_assistant.py dev
```

---

## Words That Are Ignored (When Agent is Talking)

```
yeah, yes, yep, yup, ok, okay, hmm, hm, mm, mhm, 
uh-huh, right, sure, got it, ah, uh, um, oh, 
i see, alright, all right
```

## Words That Always Interrupt

```
stop, wait, hold on, pause, no, cancel, quit, 
exit, excuse me, actually, but, however
```

---

## How to Customize

Edit `livekit-agents/livekit/agents/voice/agent_activity.py`:

```python
class BackchannelConfig:
    ignore_words: list[str] = field(default_factory=lambda: [
        "yeah", "ok", "hmm",  # <-- edit this list
    ])
    
    interrupt_words: list[str] = field(default_factory=lambda: [
        "stop", "wait", "no",  # <-- edit this list
    ])
```

---

## File Changed

`livekit-agents/livekit/agents/voice/agent_activity.py`

---

## How It Works (Simple)

```
User speaks "yeah"
      ↓
Is agent talking? 
      ↓
  YES → Is it just filler words?
            ↓
        YES → IGNORE (agent keeps talking)
        NO  → INTERRUPT (agent stops)
      ↓
  NO  → RESPOND normally
```

---

## Test It

1. Ask agent to explain something long
2. Say "yeah" or "ok" while it talks → It should keep talking
3. Say "stop" while it talks → It should stop
4. Wait for silence, then say "yeah" → It should respond

---

## Transcript showing the code now skipped the interruptions

Look for the `Ignoring backchannel while agent speaking` logs. They show when the ignore words were detected and then skipped.

```            INFO   livekit.agents   starting worker {"version": "1.3.3", "rtc-version": "1.0.23"}
             INFO   livekit.agents   preloading plugins {"packages": ["livekit.plugins.silero", "livekit.plugins.turn_detector", "av"]}
             INFO   livekit.agents   starting inference executor  
    05:26:39 INFO   livekit.agents   initializing process {"pid": 69493, "inference": true}
    05:26:49 INFO   livekit.agents   process initialized {"pid": 69493, "inference": true, "elapsed_time": 9.95}
             INFO   livekit.agents   registered worker {"agent_name": "", "id": "AW_nQqy5pLrB4t5", "url": "wss://assignment-ly639ig7.livekit.cloud", "region": "India South", "protocol": 16}
             INFO   livekit.agents   received job request {"job_id": "AJ_Us9J6pxrGMPY", "dispatch_id": "AD_LSmmU8sVJbzD", "room": "playground-44WC-Xkwi", "room_id": "RM_GQV4LLdsERnd",         
                                                          "agent_name": "", "resuming": false, "enable_recording": false}                                                                       
             DEBUG  livekit.agents   received assignment {"agent_name": "", "room_id": "RM_GQV4LLdsERnd", "room": "playground-44WC-Xkwi", "job_id": "AJ_Us9J6pxrGMPY", "dispatch_id":           
                                                         "AD_LSmmU8sVJbzD", "enable_recording": false}                                                                                          
             INFO   livekit.agents   initializing process {"pid": 69585}
             INFO   livekit.agents   process initialized {"pid": 69585, "elapsed_time": 0.12}
             DEBUG  asyncio          Using selector: EpollSelector {"pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}
             DEBUG  livekit.agents   input stream attached {"room": "playground-44WC-Xkwi", "participant": null, "source": "SOURCE_UNKNOWN", "accepted_sources": ["SOURCE_MICROPHONE"], "pid":  
                                                           69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                    
             DEBUG  livekit.agents   http_session(): creating a new httpclient ctx {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}
Nvidia Encoder is supported.
Nvidia Decoder is supported.
    05:26:51 DEBUG  livekit.agents   using audio io: `RoomIO` -> `AgentSession` -> `TranscriptSynchronizer` ->    {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY",   
                                     `RoomIO`                                                                     "room_id": "RM_GQV4LLdsERnd"}                                                 
             DEBUG  livekit.agents   using transcript io: `AgentSession` -> `TranscriptSynchronizer` -> `RoomIO` {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY",    
                                                                                                                 "room_id": "RM_GQV4LLdsERnd"}                                                  
             DEBUG  livekit.agents   start reading stream {"room": "playground-44WC-Xkwi", "participant": "identity-qX6P", "source": "SOURCE_MICROPHONE", "pid": 69585, "job_id":               
                                                          "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                      
             INFO   livekit.agents   LLM metrics {"room": "playground-44WC-Xkwi", "model_name": "openai/gpt-4.1-mini", "model_provider": "livekit", "ttft": 0.96, "prompt_tokens": 151,         
                                                 "prompt_cached_tokens": 0, "completion_tokens": 25, "tokens_per_second": 21.14, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":          
                                                 "RM_GQV4LLdsERnd"}                                                                                                                             
    05:26:53 INFO   livekit.agents   TTS metrics {"room": "playground-44WC-Xkwi", "model_name": "cartesia/sonic-2", "model_provider": "livekit", "ttfb": 0.27258383599837543, "audio_duration": 
                                                 7.48, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                 
             INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.85, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:26:55 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.85, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:00 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.95, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:03 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.85, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
             DEBUG  livekit.agents   received user transcript {"room": "playground-44WC-Xkwi", "user_transcript": "Me a really, really long story.", "language": "en", "transcript_delay":      
                                                              0.47107744216918945, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                     
             DEBUG  livekit.plugins… eou prediction {"room": "playground-44WC-Xkwi", "eou_probability": 0.03564910963177681, "duration": 0.082, "input": "<|im_start|>assistant\nhey there what 
                                                    can i do for you today want to chat need some info or just feeling curious<|im_end|>\n<|im_start|>user\nme a really really long story",     
                                                    "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                    
             DEBUG  livekit.agents   using preemptive generation {"room": "playground-44WC-Xkwi", "preemptive_lead_time": 0.16567659378051758, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY",       
                                                                 "room_id": "RM_GQV4LLdsERnd"}                                                                                                  
             INFO   livekit.agents   EOU metrics {"room": "playground-44WC-Xkwi", "model_name": "multilingual", "model_provider": "livekit", "end_of_utterance_delay": 0.64,                    
                                                 "transcription_delay": 0.47, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                          
    05:27:04 INFO   livekit.agents   LLM metrics {"room": "playground-44WC-Xkwi", "model_name": "openai/gpt-4.1-mini", "model_provider": "livekit", "ttft": 0.68, "prompt_tokens": 191,         
                                                 "prompt_cached_tokens": 0, "completion_tokens": 52, "tokens_per_second": 41.1, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":           
                                                 "RM_GQV4LLdsERnd"}                                                                                                                             
    05:27:08 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.0, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:09 INFO   livekit.agents   TTS metrics {"room": "playground-44WC-Xkwi", "model_name": "cartesia/sonic-2", "model_provider": "livekit", "ttfb": 0.32884845299849985, "audio_duration": 
                                                 14.12, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                
    05:27:10 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.55, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:15 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.0, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:18 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.55, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:21 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.1, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
             DEBUG  livekit.agents   received user transcript {"room": "playground-44WC-Xkwi", "user_transcript": "Yeah.", "language": "en", "transcript_delay": 0.5170876979827881, "pid":     
                                                              69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                 
    05:27:22 DEBUG  livekit.plugins… eou prediction {"room": "playground-44WC-Xkwi", "eou_probability": 0.01924443617463112, "duration": 0.211, "input": "<|im_start|>assistant\nhey there what 
                                                    can i do for you today want to chat need some info or just feeling curious<|im_end|>\n<|im_start|>user\nme a really really long             
                                                    story<|im_end|>\n<|im_start|>assistant\nalright brace yourself for a tale that goes on and on once upon a time actually how about i start   
                                                    with a quick spark if you like it i can keep going and twist it around in all sorts of directions sound                                     
                                                    good<|im_end|>\n<|im_start|>user\nyeah", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                           
             DEBUG  livekit.agents   using preemptive generation {"room": "playground-44WC-Xkwi", "preemptive_lead_time": 0.24658536911010742, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY",       
                                                                 "room_id": "RM_GQV4LLdsERnd"}                                                                                                  
             INFO   livekit.agents   EOU metrics {"room": "playground-44WC-Xkwi", "model_name": "multilingual", "model_provider": "livekit", "end_of_utterance_delay": 0.76,                    
                                                 "transcription_delay": 0.52, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                          
    05:27:24 INFO   livekit.agents   LLM metrics {"room": "playground-44WC-Xkwi", "model_name": "openai/gpt-4.1-mini", "model_provider": "livekit", "ttft": 0.72, "prompt_tokens": 252,         
                                                 "prompt_cached_tokens": 0, "completion_tokens": 189, "tokens_per_second": 66.05, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":         
                                                 "RM_GQV4LLdsERnd"}                                                                                                                             
    05:27:25 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.2, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:29 DEBUG  livekit.agents   Ignoring backchannel while agent speaking: 'Okay.' {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":                  
                                                                                        "RM_GQV4LLdsERnd"}                                                                                      
    05:27:30 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.4, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
             DEBUG  livekit.agents   Ignoring backchannel while agent speaking: 'Okay.' {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":                  
                                                                                        "RM_GQV4LLdsERnd"}                                                                                      
             DEBUG  livekit.agents   received user transcript {"room": "playground-44WC-Xkwi", "user_transcript": "Okay.", "language": "en", "transcript_delay": 0.5427289009094238, "pid":     
                                                              69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                 
             DEBUG  livekit.plugins… eou prediction {"room": "playground-44WC-Xkwi", "eou_probability": 0.1380336731672287, "duration": 0.238, "input": "<|im_start|>assistant\nhey there what  
                                                    can i do for you today want to chat need some info or just feeling curious<|im_end|>\n<|im_start|>user\nme a really really long             
                                                    story<|im_end|>\n<|im_start|>assistant\nalright brace yourself for a tale that goes on and on once upon a time actually how about i start   
                                                    with a quick spark if you like it i can keep going and twist it around in all sorts of directions sound                                     
                                                    good<|im_end|>\n<|im_start|>user\nyeah okay", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                      
             DEBUG  livekit.agents   Ignoring backchannel turn while agent speaking: 'Okay.' {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":             
                                                                                             "RM_GQV4LLdsERnd"}                                                                                 
    05:27:34 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.4, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
             DEBUG  livekit.agents   Ignoring backchannel while agent speaking: 'Right.' {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":                 
                                                                                         "RM_GQV4LLdsERnd"}                                                                                     
             DEBUG  livekit.agents   received user transcript {"room": "playground-44WC-Xkwi", "user_transcript": "Right.", "language": "en", "transcript_delay": 0.4661092758178711, "pid":    
                                                              69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                 
    05:27:35 DEBUG  livekit.plugins… eou prediction {"room": "playground-44WC-Xkwi", "eou_probability": 0.05378274247050285, "duration": 0.175, "input": "<|im_start|>assistant\nhey there what 
                                                    can i do for you today want to chat need some info or just feeling curious<|im_end|>\n<|im_start|>user\nme a really really long             
                                                    story<|im_end|>\n<|im_start|>assistant\nalright brace yourself for a tale that goes on and on once upon a time actually how about i start   
                                                    with a quick spark if you like it i can keep going and twist it around in all sorts of directions sound                                     
                                                    good<|im_end|>\n<|im_start|>user\nyeah okay right", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                
             DEBUG  livekit.agents   Ignoring backchannel turn while agent speaking: 'Okay. Right.' {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":      
                                                                                                    "RM_GQV4LLdsERnd"}                                                                          
    05:27:36 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.9, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:38 DEBUG  livekit.agents   Ignoring backchannel while agent speaking: 'Okay.' {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":                  
                                                                                        "RM_GQV4LLdsERnd"}                                                                                      
             INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.3, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
             DEBUG  livekit.agents   Ignoring backchannel while agent speaking: 'Got it.' {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":                
                                                                                          "RM_GQV4LLdsERnd"}                                                                                    
             DEBUG  livekit.agents   received user transcript {"room": "playground-44WC-Xkwi", "user_transcript": "Got it.", "language": "en", "transcript_delay": 0.46759510040283203, "pid":  
                                                              69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                 
    05:27:39 DEBUG  livekit.plugins… eou prediction {"room": "playground-44WC-Xkwi", "eou_probability": 0.2666323781013489, "duration": 0.154, "input": "<|im_start|>assistant\nhey there what  
                                                    can i do for you today want to chat need some info or just feeling curious<|im_end|>\n<|im_start|>user\nme a really really long             
                                                    story<|im_end|>\n<|im_start|>assistant\nalright brace yourself for a tale that goes on and on once upon a time actually how about i start   
                                                    with a quick spark if you like it i can keep going and twist it around in all sorts of directions sound                                     
                                                    good<|im_end|>\n<|im_start|>user\nyeah okay right got it", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}         
             DEBUG  livekit.agents   Ignoring backchannel turn while agent speaking: 'Okay. Right. Got it.' {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY",         
                                                                                                            "room_id": "RM_GQV4LLdsERnd"}                                                       
    05:27:40 INFO   livekit.agents   TTS metrics {"room": "playground-44WC-Xkwi", "model_name": "cartesia/sonic-2", "model_provider": "livekit", "ttfb": 0.2855382960042334, "audio_duration":  
                                                 57.59, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                
             INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.7, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:43 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.7, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:45 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.95, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:48 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.75, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:50 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.75, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
             DEBUG  livekit.agents   Ignoring backchannel while agent speaking: 'Sure.' {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":                  
                                                                                        "RM_GQV4LLdsERnd"}                                                                                      
             DEBUG  livekit.agents   received user transcript {"room": "playground-44WC-Xkwi", "user_transcript": "Sure.", "language": "en", "transcript_delay": 0.5093498229980469, "pid":     
                                                              69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                 
    05:27:51 DEBUG  livekit.plugins… eou prediction {"room": "playground-44WC-Xkwi", "eou_probability": 0.14235681295394897, "duration": 0.178, "input": "<|im_start|>assistant\nhey there what 
                                                    can i do for you today want to chat need some info or just feeling curious<|im_end|>\n<|im_start|>user\nme a really really long             
                                                    story<|im_end|>\n<|im_start|>assistant\nalright brace yourself for a tale that goes on and on once upon a time actually how about i start   
                                                    with a quick spark if you like it i can keep going and twist it around in all sorts of directions sound                                     
                                                    good<|im_end|>\n<|im_start|>user\nyeah okay right got it sure", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}    
             DEBUG  livekit.agents   Ignoring backchannel turn while agent speaking: 'Okay. Right. Got it. Sure.' {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY",   
                                                                                                                  "room_id": "RM_GQV4LLdsERnd"}                                                 
             INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.05, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:55 DEBUG  livekit.plugins… eou prediction {"room": "playground-44WC-Xkwi", "eou_probability": 0.14235681295394897, "duration": 0.154, "input": "<|im_start|>assistant\nhey there what 
                                                    can i do for you today want to chat need some info or just feeling curious<|im_end|>\n<|im_start|>user\nme a really really long             
                                                    story<|im_end|>\n<|im_start|>assistant\nalright brace yourself for a tale that goes on and on once upon a time actually how about i start   
                                                    with a quick spark if you like it i can keep going and twist it around in all sorts of directions sound                                     
                                                    good<|im_end|>\n<|im_start|>user\nyeah okay right got it sure", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}    
             DEBUG  livekit.agents   Ignoring backchannel turn while agent speaking: 'Okay. Right. Got it. Sure.' {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY",   
                                                                                                                  "room_id": "RM_GQV4LLdsERnd"}                                                 
    05:27:56 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.25, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
             DEBUG  livekit.agents   received user transcript {"room": "playground-44WC-Xkwi", "user_transcript": "Wait.", "language": "en", "transcript_delay": 0.91152024269104, "pid": 69585,
                                                              "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                        
             DEBUG  livekit.plugins… eou prediction {"room": "playground-44WC-Xkwi", "eou_probability": 0.047944359481334686, "duration": 0.243, "input": "<|im_start|>assistant\nhey there what
                                                    can i do for you today want to chat need some info or just feeling curious<|im_end|>\n<|im_start|>user\nme a really really long             
                                                    story<|im_end|>\n<|im_start|>assistant\nalright brace yourself for a tale that goes on and on once upon a time actually how about i start   
                                                    with a quick spark if you like it i can keep going and twist it around in all sorts of directions sound                                     
                                                    good<|im_end|>\n<|im_start|>user\nyeah okay right got it sure wait", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":                  
                                                    "RM_GQV4LLdsERnd"}                                                                                                                          
             INFO   livekit.agents   EOU metrics {"room": "playground-44WC-Xkwi", "model_name": "multilingual", "model_provider": "livekit", "end_of_utterance_delay": 1.16,                    
                                                 "transcription_delay": 0.91, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                          
    05:27:57 INFO   livekit.agents   LLM metrics {"room": "playground-44WC-Xkwi", "model_name": "openai/gpt-4.1-mini", "model_provider": "livekit", "ttft": 0.72, "prompt_tokens": 379,         
                                                 "prompt_cached_tokens": 0, "completion_tokens": 30, "tokens_per_second": 25.37, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":          
                                                 "RM_GQV4LLdsERnd"}                                                                                                                             
    05:27:58 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.25, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:27:59 INFO   livekit.agents   TTS metrics {"room": "playground-44WC-Xkwi", "model_name": "cartesia/sonic-2", "model_provider": "livekit", "ttfb": 0.2693334230061737, "audio_duration":  
                                                 7.06, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                 
    05:28:01 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.0, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:05 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.3, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:10 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.45, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:15 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.2, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:18 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.85, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:23 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.85, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:26 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.15, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:30 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.7, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:31 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 0.5, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:36 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.9, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:40 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.05, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:44 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.5, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:49 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.85, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:53 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.9, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:28:58 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.55, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:02 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.1, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:06 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.5, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:09 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.15, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:14 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.2, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:18 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.1, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:23 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.85, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:27 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.1, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:31 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.8, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:35 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.65, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:39 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.85, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:44 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.9, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
             INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 0.15, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:47 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.6, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
             INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 0.4, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:48 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.4, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:49 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 0.8, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:51 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.95, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:52 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 0.85, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:53 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 0.55, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:29:58 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.0, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:00 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.45, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:01 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.3, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:02 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.05, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:05 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.55, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:10 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.95, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:11 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.35, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:13 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.35, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:18 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.0, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:21 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.85, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:25 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.0, "pid": 69585, "job_id":  
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:29 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.55, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:34 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.95, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:37 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.05, "pid": 69585, "job_id": 
                                                 "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                                               
    05:30:42 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.95, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:30:47 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.95, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:30:51 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.85, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:30:54 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.6, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:30:58 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.15, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:02 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.6, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:03 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.15, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
             DEBUG  livekit.agents   received user transcript {"room": "playground-44WC-Xkwi", "user_transcript": "Yeah.", "language": "en", "transcript_delay": 0.6305022239685059,      
                                                              "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                    
             DEBUG  livekit.plugins… eou prediction {"room": "playground-44WC-Xkwi", "eou_probability": 0.017827074974775314, "duration": 0.215, "input": "<|im_start|>assistant\nalright 
                                                    brace yourself for a tale that goes on and on once upon a time actually how about i start with a quick spark if you like it i can keep
                                                    going and twist it around in all sorts of directions sound                                                                            
                                                    good<|im_end|>\n<|im_start|>user\nyeah<|im_end|>\n<|im_start|>assistant\nokay here we go once upon a time in a tiny village nestled   
                                                    between giant mountains there lived a baker named toby who dreamed of making the world's most magical bread every day he'd mix secret 
                                                    ingredients whispered to him by the wind hoping to bake a loaf that could sing one morning he woke up to find a mysterious glowing    
                                                    grain on his doorstep curious toby baked it into bread and when he took a bite he heard the faintest melodylike a lullaby from the    
                                                    mountain spirits excited<|im_end|>\n<|im_start|>user\nokay right got it sure wait<|im_end|>\n<|im_start|>assistant\ntake your time no 
                                                    rush at all let me know when you're ready to hear what happens next or if you want me to switch                                       
                                                    gears<|im_end|>\n<|im_start|>user\nyeah", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                    
             DEBUG  livekit.agents   using preemptive generation {"room": "playground-44WC-Xkwi", "preemptive_lead_time": 0.22072577476501465, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", 
                                                                 "room_id": "RM_GQV4LLdsERnd"}                                                                                            
             INFO   livekit.agents   EOU metrics {"room": "playground-44WC-Xkwi", "model_name": "multilingual", "model_provider": "livekit", "end_of_utterance_delay": 0.85,              
                                                 "transcription_delay": 0.63, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                    
    05:31:06 DEBUG  livekit.agents   flush audio emitter due to slow audio generation {"room": "playground-44WC-Xkwi", "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":              
                                                                                      "RM_GQV4LLdsERnd"}                                                                                  
             INFO   livekit.agents   LLM metrics {"room": "playground-44WC-Xkwi", "model_name": "openai/gpt-4.1-mini", "model_provider": "livekit", "ttft": 1.07, "prompt_tokens": 418,   
                                                 "prompt_cached_tokens": 0, "completion_tokens": 110, "tokens_per_second": 36.45, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":   
                                                 "RM_GQV4LLdsERnd"}                                                                                                                       
    05:31:08 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.95, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:11 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.05, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:14 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.75, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
             DEBUG  livekit.agents   received user transcript {"room": "playground-44WC-Xkwi", "user_transcript": "So", "language": "en", "transcript_delay": 1.0929808616638184, "pid":  
                                                              69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                           
             DEBUG  livekit.plugins… eou prediction {"room": "playground-44WC-Xkwi", "eou_probability": 0.0023743619676679373, "duration": 0.23, "input":                                 
                                                    "<|im_start|>user\nyeah<|im_end|>\n<|im_start|>assistant\nokay here we go once upon a time in a tiny village nestled between giant    
                                                    mountains there lived a baker named toby who dreamed of making the world's most magical bread every day he'd mix secret ingredients   
                                                    whispered to him by the wind hoping to bake a loaf that could sing one morning he woke up to find a mysterious glowing grain on his   
                                                    doorstep curious toby baked it into bread and when he took a bite he heard the faintest melodylike a lullaby from the mountain spirits
                                                    excited<|im_end|>\n<|im_start|>user\nokay right got it sure wait<|im_end|>\n<|im_start|>assistant\ntake your time no rush at all let  
                                                    me know when you're ready to hear what happens next or if you want me to switch gears<|im_end|>\n<|im_start|>user\nyeah so", "pid":   
                                                    69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                     
    05:31:16 INFO   livekit.agents   EOU metrics {"room": "playground-44WC-Xkwi", "model_name": "multilingual", "model_provider": "livekit", "end_of_utterance_delay": 3.0,               
                                                 "transcription_delay": 1.09, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                    
             INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.45, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:18 INFO   livekit.agents   LLM metrics {"room": "playground-44WC-Xkwi", "model_name": "openai/gpt-4.1-mini", "model_provider": "livekit", "ttft": 0.67, "prompt_tokens": 453,   
                                                 "prompt_cached_tokens": 0, "completion_tokens": 70, "tokens_per_second": 37.43, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id":    
                                                 "RM_GQV4LLdsERnd"}                                                                                                                       
    05:31:21 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.95, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:22 INFO   livekit.agents   TTS metrics {"room": "playground-44WC-Xkwi", "model_name": "cartesia/sonic-2", "model_provider": "livekit", "ttfb": 0.2640098530027899,              
                                                 "audio_duration": 19.18, "pid": 69585, "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                        
    05:31:23 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.7, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:28 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.35, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:29 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.05, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:33 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.9, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:38 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.0, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:39 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 0.25, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:44 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.0, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:48 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.75, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:53 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.65, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:31:57 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.15, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:00 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.35, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:05 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.0, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:09 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.2, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:12 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.8, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:15 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.2, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:20 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.0, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:21 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.15, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:23 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.0, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:24 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 0.95, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
             INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 0.6, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:25 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 0.45, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:30 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.95, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:33 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.05, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:37 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.8, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:40 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.0, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:43 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.45, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:45 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.35, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:48 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.3, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:49 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 1.3, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:54 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.75, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:32:58 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.9, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:02 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.35, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:03 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 0.5, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:08 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.0, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:11 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.85, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:16 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.9, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:20 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.6, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:25 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.65, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:28 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.0, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:32 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.75, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:37 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.85, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:40 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.0, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:44 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.95, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:48 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.7, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:51 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.3, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:33:56 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.9, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:01 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.85, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:06 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.0, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:10 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.35, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:14 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.55, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:17 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.35, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:20 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.05, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:23 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.35, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:27 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.15, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:31 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.3, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:35 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.95, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:38 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.4, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:43 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.25, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:45 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.45, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:50 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.95, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:55 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 5.0, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:34:57 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 2.1, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:35:02 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.95, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:35:06 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.05, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:35:10 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.4, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:35:14 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.8, "pid": 69585,      
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:35:18 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.95, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:35:22 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.25, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:35:27 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 4.95, "pid": 69585,     
                                                 "job_id": "AJ_Us9J6pxrGMPY", "room_id": "RM_GQV4LLdsERnd"}                                                                               
    05:35:30 INFO   livekit.agents   STT metrics {"room": "playground-44WC-Xkwi", "model_name": "deepgram/nova-3", "model_provider": "livekit", "audio_duration": 3.05, "pid": 69585,     

```