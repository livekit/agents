[1mdiff --git a/livekit-agents/livekit/agents/voice/agent_activity.py b/livekit-agents/livekit/agents/voice/agent_activity.py[m
[1mindex 43d4e71e..ee2a3c16 100644[m
[1m--- a/livekit-agents/livekit/agents/voice/agent_activity.py[m
[1m+++ b/livekit-agents/livekit/agents/voice/agent_activity.py[m
[36m@@ -2,6 +2,11 @@[m [mfrom __future__ import annotations[m
 from agents.extensions.interrupt_handler import InterruptFilter, livekit_asr_adapter[m
 [m
 import asyncio[m
[32m+[m[32mtry:[m
[32m+[m[32m    from agents.extensions.interrupt_handler import InterruptFilter[m
[32m+[m[32mexcept Exception:[m
[32m+[m[32m    InterruptFilter = None[m
[32m+[m
 import contextvars[m
 import heapq[m
 import json[m
[36m@@ -88,6 +93,9 @@[m [mclass _PreemptiveGeneration:[m
 # NOTE: AgentActivity isn't exposed to the public API[m
 class AgentActivity(RecognitionHooks):[m
     def __init__(self, agent: Agent, sess: AgentSession) -> None:[m
[32m+[m[32m        self._interrupt_requested = False[m
[32m+[m[32m        self._interrupt_filter = InterruptFilter() if InterruptFilter is not None else None[m
[32m+[m
         self._interrupt_filter = InterruptFilter([m
             on_interrupt=lambda info: asyncio.create_task(self._handle_interrupt(info))[m
         )[m
[36m@@ -210,6 +218,20 @@[m [mclass AgentActivity(RecognitionHooks):[m
         # speeches that audio playout finished but not done because of tool calls[m
         self._background_speeches: set[SpeechHandle] = set()[m
 [m
[32m+[m[32m    async def _maybe_handle_interrupt(self, text: str, *, confidence: float = 1.0, agent_speaking: bool = False) -> None:[m
[32m+[m[32m        try:[m
[32m+[m[32m            result = await self._interrupt_filter.on_asr_event(text, confidence=confidence, agent_speaking=agent_speaking)[m
[32m+[m[32m        except Exception:[m
[32m+[m[32m            import logging[m
[32m+[m[32m            logging.exception("InterruptFilter failed")[m
[32m+[m[32m            return[m
[32m+[m[32m        if result.get("should_stop_agent"):[m
[32m+[m[32m            # request session-level interrupt (the session shim we added earlier)[m
[32m+[m[32m            try:[m
[32m+[m[32m                self._session.request_interrupt()[m
[32m+[m[32m            except Exception:[m
[32m+[m[32m                logging.exception("Failed to request session interrupt")[m
[32m+[m
     async def _handle_interrupt(self, info: dict):[m
         try:[m
             if hasattr(self, "interrupt"):[m
[36m@@ -222,6 +244,16 @@[m [mclass AgentActivity(RecognitionHooks):[m
         except Exception:[m
             if hasattr(self, "logger"):[m
                 self.logger.exception("Error handling interrupt")[m
[32m+[m[41m    [m
[32m+[m[32m    def request_interrupt(self) -> None:[m
[32m+[m[32m        self._interrupt_requested = True[m
[32m+[m
[32m+[m[32m    def consume_interrupt_request(self) -> bool:[m
[32m+[m[32m        if self._interrupt_requested:[m
[32m+[m[32m            self._interrupt_requested = False[m
[32m+[m[32m            return True[m
[32m+[m[32m        return False[m
[32m+[m
 [m
     @property[m
     def scheduling_paused(self) -> bool:[m
[36m@@ -934,6 +966,10 @@[m [mclass AgentActivity(RecognitionHooks):[m
             self._rt_session.clear_audio()[m
 [m
     def commit_user_turn(self, *, transcript_timeout: float, stt_flush_duration: float) -> None:[m
[32m+[m[32m        if self.consume_interrupt_request():[m
[32m+[m[32m            self._logger.info("Interrupt requested — stopping agent output early")[m
[32m+[m[32m            asyncio.create_task(self._stop_speaking_now())[m
[32m+[m
         assert self._audio_recognition is not None[m
         self._audio_recognition.commit_user_turn([m
             audio_detached=not self._session.input.audio_enabled,[m
[36m@@ -1211,6 +1247,9 @@[m [mclass AgentActivity(RecognitionHooks):[m
             ),[m
         )[m
 [m
[32m+[m[32m        if self._interrupt_filter is not None:[m
[32m+[m[32m            asyncio.create_task(self._maybe_handle_interrupt(ev.alternatives[0].text, confidence=getattr(ev.alternatives[0],'confidence',1.0), agent_speaking=self._session.use_tts_aligned_transcript() if hasattr(self._session,'use_tts_aligned_transcript') else True))[m
[32m+[m
         try:[m
             raw_text = getattr(ev.alternatives[0], "text", "") or ""[m
             raw_conf = getattr(ev.alternatives[0], "confidence", None)[m
[36m@@ -1258,6 +1297,9 @@[m [mclass AgentActivity(RecognitionHooks):[m
                 speaker_id=ev.alternatives[0].speaker_id,[m
             ),[m
         )[m
[32m+[m
[32m+[m[32m        if self._interrupt_filter is not None:[m
[32m+[m[32m            asyncio.create_task(self._maybe_handle_interrupt(ev.alternatives[0].text, confidence=getattr(ev.alternatives[0],'confidence',1.0), agent_speaking=self._session.use_tts_aligned_transcript() if hasattr(self._session,'use_tts_aligned_transcript') else True))[m
         [m
         try:[m
             raw_text = getattr(ev.alternatives[0], "text", "") or ""[m
