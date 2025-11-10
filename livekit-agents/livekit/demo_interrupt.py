# demo_interrupt.py
from livekit.extension.interrupt_handler import InterruptHandler, ASRChunk, InterruptDecision

ih = InterruptHandler()  # default config
print("TTS start (agent speaking)")
ih.on_tts_start()

tests = [
    ("umm hmm", 0.9),
    ("wait one second", 0.3),
    ("ok got it", 0.95),
    ("stop please", 0.8),
    ("umm umm umm hmm umm umm hmm", 0.8)
]

for text, conf in tests:
    decision, meta = ih.classify(ASRChunk(text=text, avg_confidence=conf, is_final=True))
    print(f"{text!r:>18} -> {decision} [{meta['reason']}]")

print("TTS end (agent quiet)")
ih.on_tts_end()
text, conf = "umm", 0.2
decision, meta = ih.classify(ASRChunk(text=text, avg_confidence=conf, is_final=True))
print(f"{text!r:>18} -> {decision} [{meta['reason']}]")
