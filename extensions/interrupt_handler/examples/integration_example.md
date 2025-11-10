# How to wire this into LiveKit Agents (high-level)

1. In your LiveKit agent code (where it subscribes to transcription or ASR events),
   create an instance of InterruptHandler.

   from livekit_interrupt_handler import InterruptHandler
   handler = InterruptHandler()

2. Whenever LiveKit emits an event (pseudo-code):
   def on_transcription_event(asr_payload): # asr_payload should include 'text' and if available 'confidence' (0..1)
   result = await handler.on_transcription({
   "text": asr_payload.text,
   "confidence": asr_payload.confidence
   })
   if result['action'] == 'interrupt': # invoke whatever API you use to stop TTS / pause agent
   agent.stop_speaking() # or agent.pause_tts()
   elif result['action'] == 'register': # treat as user utterance: feed into dialogue manager
   dialogue_manager.on_user_utterance(result['text'])
   elif result['action'] == 'ignore': # do nothing
   pass

3. Maintain agent speaking state:

   - when agent starts TTS: await handler.set_agent_speaking(True)
   - when agent finishes/pauses: await handler.set_agent_speaking(False)

4. Do NOT change LiveKit VAD internals. This is an extension-layer filter on top of ASR events.
