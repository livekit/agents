# examples/interrupt_handler_simple.py
# A minimal interruption handler (no livekit dependencies)

FILLER_WORDS = {"uh", "um", "umm", "hmm", "mm", "haan"}
COMMAND_KEYWORDS = {"stop", "wait", "no", "cancel"}

class SimpleInterruptionHandler:
    def __init__(self):
        self.agent_is_speaking = False

    async def stop_speaking(self):
        print("[HANDLER] stop_speaking() called -> agent should stop now")
        self.agent_is_speaking = False

    async def handle_transcript(self, transcript: str, is_final: bool = True):
        text = transcript.strip().lower()
        if not is_final:
            print("[HANDLER] interim -> ignoring:", text)
            return

        words = text.split()
        if not words:
            print("[HANDLER] empty transcript -> ignoring")
            return

        wordset = set(words)

        if self.agent_is_speaking:
            # filler-only while agent is speaking -> ignore
            if wordset.issubset(FILLER_WORDS):
                print("[IGNORED FILLER WHILE SPEAKING]:", transcript)
                return

            # contains command keyword -> interrupt
            if COMMAND_KEYWORDS & wordset:
                print("[INTERRUPTION COMMAND]:", transcript)
                await self.stop_speaking()
                return

            # otherwise unknown while speaking
            print("[UNKNOWN/OTHER WHILE SPEAKING]:", transcript)
        else:
            # agent idle -> register user input
            print("[USER INPUT WHILE IDLE]:", transcript)
