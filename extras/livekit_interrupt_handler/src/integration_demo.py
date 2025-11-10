import asyncio
from filler_filter import FillerFilter, ASRResult

async def main():
    # Initialize filter with default filler words
    ff = FillerFilter(ignored_words=['uh','umm','hmm','haan'], confidence_threshold=0.6)

    # Define simple callbacks
    ff.on_suppress = lambda asr: print(f"[SUPPRESS] {asr.text} (conf={asr.confidence})")
    ff.on_forward  = lambda asr: print(f"[FORWARD] {asr.text} (conf={asr.confidence})")
    ff.on_register = lambda asr: print(f"[REGISTER] {asr.text} (conf={asr.confidence})")

    async def simulate(agent_speaking, events):
        ff.set_agent_speaking(agent_speaking)
        print(f"\n=== Agent speaking = {agent_speaking} ===")
        for evt in events:
            await ff.handle_asr(ASRResult(
                text=evt['text'],
                confidence=evt.get('conf', 0.9),
                is_final=evt.get('final', True)
            ))
            await asyncio.sleep(evt.get('delay', 0.05))
        await asyncio.sleep(0.3)  # wait for flush

    # Example scenarios
    scenarios = [
        (True,  [{'text': 'uh', 'conf': 0.95}]),
        (True,  [{'text': 'wait one second', 'conf': 0.95}]),
        (False, [{'text': 'umm', 'conf': 0.95}]),
        (True,  [{'text': 'umm okay stop', 'conf': 0.95}]),
        (True,  [{'text': 'hmm yeah', 'conf': 0.3}]),
    ]

    for agent_speaking, events in scenarios:
        await simulate(agent_speaking, events)

if __name__ == "__main__":
    asyncio.run(main())
