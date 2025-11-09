# import asyncio
# import re
# from typing import Optional

# from livekit.agents import (
#     AgentSession,
#     AgentStateChangedEvent,
#     UserInputTranscribedEvent,
# )

# from .config import InterruptConfig
# from .logging_utils import get_logger

# logger = get_logger("controller")
# #Filler words set 
# # filler_words = {'uh', 'um', 'hmm', 'like', 'you know', 'so', 'actually', 'basically', 'right', 'i mean'}

# class FillerAwareInterruptController:
#     """
#     Filler-aware extension on top of AgentSession.

#     - Does NOT modify VAD.
#     - Uses only events + session.interrupt().
#     """

#     def __init__(self, session: AgentSession, config: Optional[InterruptConfig] = None):
#         self.session = session
#         self.cfg = config or InterruptConfig()

#         self._agent_speaking: bool = False
#         self._lock = asyncio.Lock()

#         self._attach_event_handlers()

#         if self.cfg.debug_logging:
#             logger.info(
#                 "FillerAwareInterruptController initialized with "
#                 f"{len(self.cfg.filler_words)} filler words and "
#                 f"{len(self.cfg.hard_interrupt_phrases)} hard interrupt phrases."
#             )

#     @property
#     def agent_speaking(self) -> bool:
#         return self._agent_speaking

#     # ---------------- Event Wiring ----------------

#     def _attach_event_handlers(self) -> None:
#         @self.session.on("agent_state_changed")
#         def _on_agent_state_changed(ev: AgentStateChangedEvent):
#             self._agent_speaking = (ev.new_state == "speaking")
#             if self.cfg.debug_logging:
#                 logger.debug(
#                     "agent_state_changed: %s -> %s (speaking=%s)",
#                     ev.old_state,
#                     ev.new_state,
#                     self._agent_speaking,
#                 )

#         # @self.session.on("user_input_transcribed")
#         # async def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
#         #     await self._handle_transcription_event(ev)
#         @self.session.on("user_input_transcribed")
#         def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
#             # LiveKit's event emitter expects sync callbacks.
#             # We delegate to the async handler via create_task.
#             asyncio.create_task(self._handle_transcription_event(ev))

#     # ---------------- Core Handler ----------------

#     async def _handle_transcription_event(self, ev: UserInputTranscribedEvent) -> None:
#         text = (ev.transcript or "").strip().lower()
#         if not text:
#             return

#         confidence = getattr(ev, "confidence", None)

#         async with self._lock:
#             # 1) Drop low-confidence input if provided
#             if confidence is not None and confidence < self.cfg.min_confidence:
#                 if self.cfg.debug_logging:
#                     logger.debug(
#                         "ignored low-confidence: '%s' (%.3f)", text, confidence
#                     )
#                 return

#             # 2) If agent not speaking → no special filter
#             if not self._agent_speaking:
#                 if self.cfg.debug_logging:
#                     logger.debug(
#                         "agent not speaking; letting input pass through: '%s'", text
#                     )
#                 return

#             # 3) Agent is speaking → classify this as filler/interrupt
#             decision = self._classify_input(text, is_final=ev.is_final)

#             if decision == "FILLER":
#                 if self.cfg.debug_logging:
#                     logger.info("ignored filler during speech: '%s'", text)
#                 return

#             if decision == "PENDING":
#                 if self.cfg.debug_logging:
#                     logger.debug("pending more context: '%s'", text)
#                 return

#             if decision == "HARD_INTERRUPT":
#                 await self._interrupt_agent(f"hard interrupt phrase in '{text}'")
#                 return

#             if decision == "REAL_INTERRUPT":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Stop":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Wait":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Hold on":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "One second":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Listen":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Bas":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Ruk":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Ruko":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Thoda ruk":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "No not that one":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return

#     # ---------------- Classification ----------------

#     def _classify_input(self, text: str, is_final: bool) -> str:
#         """
#         Returns:
#           - "FILLER"
#           - "PENDING"
#           - "HARD_INTERRUPT"
#           - "REAL_INTERRUPT"
#         """
#         tokens = [t for t in re.findall(r"\w+", text) if t]

#         if not tokens:
#             return "FILLER"

#         # pure filler: all tokens in filler set & short
#         if (
#             len(tokens) <= self.cfg.max_filler_tokens
#             and all(t in self.cfg.filler_words for t in tokens)
#         ):
#             return "FILLER"

#         # hard interrupt phrase substring match
#         for phrase in self.cfg.hard_interrupt_phrases:
#             if phrase in text:
#                 return "HARD_INTERRUPT"

#         non_filler = [t for t in tokens if t not in self.cfg.filler_words]

#         # If it's a tiny partial & not final → wait
#         if len(non_filler) < self.cfg.min_real_words_to_interrupt and not is_final:
#             return "PENDING"

#         # Looks meaningful enough → treat as real interruption
#         return "REAL_INTERRUPT"

#     # ---------------- Interrupt ----------------

#     async def _interrupt_agent(self, reason: str) -> None:
#         if self.cfg.debug_logging:
#             logger.info("Interrupting agent: %s", reason)
#         try:
#             await self.session.interrupt()
#         except Exception as e:
#             logger.error("Failed to interrupt agent: %s", e)
# import asyncio
# import re
# from typing import Optional

# from livekit.agents import (
#     AgentSession,
#     AgentStateChangedEvent,
#     UserInputTranscribedEvent,
# )

# from .config import InterruptConfig
# from .logging_utils import get_logger

# logger = get_logger("controller")
# #Filler words set 
# # filler_words = {'uh', 'um', 'hmm', 'like', 'you know', 'so', 'actually', 'basically', 'right', 'i mean'}

# class FillerAwareInterruptController:
#     """
#     Filler-aware extension on top of AgentSession.

#     - Does NOT modify VAD.
#     - Uses only events + session.interrupt().
#     """

#     def __init__(self, session: AgentSession, config: Optional[InterruptConfig] = None):
#         self.session = session
#         self.cfg = config or InterruptConfig()

#         self._agent_speaking: bool = False
#         self._lock = asyncio.Lock()

#         self._attach_event_handlers()

#         if self.cfg.debug_logging:
#             logger.info(
#                 "FillerAwareInterruptController initialized with "
#                 f"{len(self.cfg.filler_words)} filler words and "
#                 f"{len(self.cfg.hard_interrupt_phrases)} hard interrupt phrases."
#             )

#     @property
#     def agent_speaking(self) -> bool:
#         return self._agent_speaking

#     # ---------------- Event Wiring ----------------

#     def _attach_event_handlers(self) -> None:
#         @self.session.on("agent_state_changed")
#         def _on_agent_state_changed(ev: AgentStateChangedEvent):
#             self._agent_speaking = (ev.new_state == "speaking")
#             if self.cfg.debug_logging:
#                 logger.debug(
#                     "agent_state_changed: %s -> %s (speaking=%s)",
#                     ev.old_state,
#                     ev.new_state,
#                     self._agent_speaking,
#                 )

#         # @self.session.on("user_input_transcribed")
#         # async def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
#         #     await self._handle_transcription_event(ev)
#         @self.session.on("user_input_transcribed")
#         def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
#             # LiveKit's event emitter expects sync callbacks.
#             # We delegate to the async handler via create_task.
#             asyncio.create_task(self._handle_transcription_event(ev))

#     # ---------------- Core Handler ----------------

#     async def _handle_transcription_event(self, ev: UserInputTranscribedEvent) -> None:
#         text = (ev.transcript or "").strip().lower()
#         if not text:
#             return

#         confidence = getattr(ev, "confidence", None)

#         async with self._lock:
#             # 1) Drop low-confidence input if provided
#             if confidence is not None and confidence < self.cfg.min_confidence:
#                 if self.cfg.debug_logging:
#                     logger.debug(
#                         "ignored low-confidence: '%s' (%.3f)", text, confidence
#                     )
#                 return

#             # 2) If agent not speaking → no special filter
#             if not self._agent_speaking:
#                 if self.cfg.debug_logging:
#                     logger.debug(
#                         "agent not speaking; letting input pass through: '%s'", text
#                     )
#                 return

#             # 3) Agent is speaking → classify this as filler/interrupt
#             decision = self._classify_input(text, is_final=ev.is_final)

#             if decision == "FILLER":
#                 if self.cfg.debug_logging:
#                     logger.info("ignored filler during speech: '%s'", text)
#                 return

#             if decision == "PENDING":
#                 if self.cfg.debug_logging:
#                     logger.debug("pending more context: '%s'", text)
#                 return

#             if decision == "HARD_INTERRUPT":
#                 await self._interrupt_agent(f"hard interrupt phrase in '{text}'")
#                 return

#             if decision == "REAL_INTERRUPT":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Stop":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Wait":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Hold on":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "One second":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Listen":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Bas":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Ruk":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Ruko":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "Thoda ruk":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return
#             if decision == "No not that one":
#                 await self._interrupt_agent(f"real interruption: '{text}'")
#                 return

#     # ---------------- Classification ----------------

#     def _classify_input(self, text: str, is_final: bool) -> str:
#         """
#         Returns:
#           - "FILLER"
#           - "PENDING"
#           - "HARD_INTERRUPT"
#           - "REAL_INTERRUPT"
#         """
#         tokens = [t for t in re.findall(r"\w+", text) if t]

#         if not tokens:
#             return "FILLER"

#         # pure filler: all tokens in filler set & short
#         if (
#             len(tokens) <= self.cfg.max_filler_tokens
#             and all(t in self.cfg.filler_words for t in tokens)
#         ):
#             return "FILLER"

#         # hard interrupt phrase substring match
#         for phrase in self.cfg.hard_interrupt_phrases:
#             if phrase in text:
#                 return "HARD_INTERRUPT"

#         non_filler = [t for t in tokens if t not in self.cfg.filler_words]

#         # If it's a tiny partial & not final → wait
#         if len(non_filler) < self.cfg.min_real_words_to_interrupt and not is_final:
#             return "PENDING"

#         # Looks meaningful enough → treat as real interruption
#         return "REAL_INTERRUPT"

#     # ---------------- Interrupt ----------------

#     async def _interrupt_agent(self, reason: str) -> None:
#         if self.cfg.debug_logging:
#             logger.info("Interrupting agent: %s", reason)
#         try:
#             await self.session.interrupt()
#         except Exception as e:
#             logger.error("Failed to interrupt agent: %s", e)
import asyncio
import re
from typing import Optional, List

from livekit.agents import (
    AgentSession,
    AgentStateChangedEvent,
    UserInputTranscribedEvent,
)

from .config import InterruptConfig
from .logging_utils import get_logger

logger = get_logger("controller")


class FillerAwareInterruptController:
    """
    Filler-aware extension on top of AgentSession.

    Core behavior:
    - While agent is SPEAKING:
        * Ignore pure fillers (from cfg.filler_words).
        * Ignore low-confidence mumbling.
        * IMMEDIATELY interrupt when:
            - any explicit command phrase appears
              (e.g. "stop", "wait", "hold on", "one second",
               "listen", "bas", "ruk", "ruko", "thoda ruk",
               "no not that one", etc. from cfg.hard_interrupt_phrases), OR
            - there is enough non-filler content (real interruption).
    - While agent is NOT speaking:
        * Never suppress — even "umm"/"hmm" are passed through as normal input.
    - No changes to base VAD, only uses events + session.interrupt().
    """
        # -------- BONUS FEATURE: Runtime command refresh --------
    def refresh_hard_commands(self) -> None:
        """Rebuild internal command caches after dynamic updates."""
        self._single_word_commands.clear()
        self._multi_word_commands.clear()

        for phrase in self.cfg.hard_interrupt_phrases:
            p = phrase.strip().lower()
            if not p:
                continue
            if " " in p:
                self._multi_word_commands.append(p)
            else:
                self._single_word_commands.append(p)

        if self.cfg.debug_logging:
            logger.info(
                "Runtime command refresh complete "
                f"(single={len(self._single_word_commands)}, "
                f"multi={len(self._multi_word_commands)})"
            )

    def __init__(self, session: AgentSession, config: Optional[InterruptConfig] = None):
        self.session = session
        self.cfg = config or InterruptConfig()

        self._agent_speaking: bool = False
        self._lock = asyncio.Lock()

        self._attach_event_handlers()

        # Precompute explicit command variants from config for fast, explicit checks
        # self._single_word_commands: List[str] = []
        # self._multi_word_commands: List[str] = []
        self._single_word_commands: List[str] = []
        self._multi_word_commands: List[str] = []
        self.refresh_hard_commands()


        for phrase in self.cfg.hard_interrupt_phrases:
            p = phrase.strip().lower()
            if not p:
                continue
            if " " in p:
                self._multi_word_commands.append(p)
            else:
                self._single_word_commands.append(p)

        if self.cfg.debug_logging:
            logger.info(
                "FillerAwareInterruptController initialized "
                f"(filler_words={len(self.cfg.filler_words)}, "
                f"hard_phrases={len(self.cfg.hard_interrupt_phrases)})"
            )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def agent_speaking(self) -> bool:
        return self._agent_speaking

    # -------------------------------------------------------------------------
    # Event wiring
    # -------------------------------------------------------------------------

    def _attach_event_handlers(self) -> None:
        @self.session.on("agent_state_changed")
        def _on_agent_state_changed(ev: AgentStateChangedEvent):
            self._agent_speaking = (ev.new_state == "speaking")
            if self.cfg.debug_logging:
                logger.debug(
                    "agent_state_changed: %s -> %s (speaking=%s)",
                    ev.old_state,
                    ev.new_state,
                    self._agent_speaking,
                )

        @self.session.on("user_input_transcribed")
        def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
            # LiveKit expects sync handlers; schedule async work separately.
            asyncio.create_task(self._handle_transcription_event(ev))

    # -------------------------------------------------------------------------
    # Core handler
    # -------------------------------------------------------------------------

    async def _handle_transcription_event(self, ev: UserInputTranscribedEvent) -> None:
        text = (ev.transcript or "").strip().lower()
        if not text:
            return

        confidence = getattr(ev, "confidence", None)
        is_final = getattr(ev, "is_final", True)

        async with self._lock:
            # 1) Filter obvious garbage: low-confidence chunks
            if confidence is not None and confidence < self.cfg.min_confidence:
                if self.cfg.debug_logging:
                    logger.debug(
                        "ignored low-confidence chunk: '%s' (%.3f)", text, confidence
                    )
                return

            # 2) If agent is NOT speaking -> do not filter.
            #    This satisfies: "Register those same words as valid speech when quiet."
            if not self._agent_speaking:
                if self.cfg.debug_logging:
                    logger.debug(
                        "agent not speaking; letting input pass through: '%s'", text
                    )
                return

            # 3) Agent IS speaking -> classify for filler vs interruption
            decision = self._classify_input(text, is_final=is_final)

            if decision == "FILLER":
                if self.cfg.debug_logging:
                    logger.info("ignored filler during speech: '%s'", text)
                return

            if decision == "PENDING":
                if self.cfg.debug_logging:
                    logger.debug("pending more context (no interrupt yet): '%s'", text)
                return

            if decision == "HARD_INTERRUPT":
                # Your explicit words & phrases land here.
                await self._interrupt_agent(
                    f"hard interrupt phrase detected in: '{text}'"
                )
                return

            if decision == "REAL_INTERRUPT":
                await self._interrupt_agent(
                    f"real interruption detected from user: '{text}'"
                )
                return

    # -------------------------------------------------------------------------
    # Classification logic
    # -------------------------------------------------------------------------

    def _classify_input(self, text: str, is_final: bool) -> str:
        """
        Classify input WHILE agent is speaking.

        Returns:
          - "FILLER"         : short, pure filler -> ignore.
          - "PENDING"        : partial fragment, wait for more.
          - "HARD_INTERRUPT" : explicit commands ("stop", "wait", etc.).
          - "REAL_INTERRUPT" : enough non-filler content -> stop.
        """
        tokens: List[str] = [t for t in re.findall(r"\w+", text) if t]

        if not tokens:
            return "FILLER"

        # 1) Pure filler check:
        #    All tokens in filler set AND within max token length.
        if (
            len(tokens) <= self.cfg.max_filler_tokens
            and all(t in self.cfg.filler_words for t in tokens)
        ):
            return "FILLER"

        # 2) Explicit HARD commands (your extra phrases), handled explicitly.

        # 2a) Single-word commands like: stop, wait, listen, bas, ruk, ruko, ...
        #     If any of these appear as a token -> immediate HARD_INTERRUPT.
        for cmd in self._single_word_commands:
            if cmd in tokens:
                return "HARD_INTERRUPT"

        # 2b) Multi-word commands like:
        #     "hold on", "one second", "thoda ruk", "no not that one"
        #     Substring match on normalized text.
        for phrase in self._multi_word_commands:
            if phrase in text:
                return "HARD_INTERRUPT"

        # 3) Count non-filler tokens for "real" interruption
        non_filler = [t for t in tokens if t not in self.cfg.filler_words]

        # If it's a tiny fragment and not final -> don't cut speech yet
        if len(non_filler) < self.cfg.min_real_words_to_interrupt and not is_final:
            return "PENDING"

        # 4) Otherwise, treat as meaningful interruption
        return "REAL_INTERRUPT"

    # -------------------------------------------------------------------------
    # Interrupt helper
    # -------------------------------------------------------------------------

    async def _interrupt_agent(self, reason: str) -> None:
        if self.cfg.debug_logging:
            logger.info("Interrupting agent due to: %s", reason)
        try:
            await self.session.interrupt()
        except Exception as e:
            logger.error("Failed to interrupt agent: %s", e)