import asyncio
from extensions.interrupt_handler.handler import decide_and_handle

async def fake_interrupt_cb(transcript, confidence):
    print(">>> INTERRUPT CB CALLED with:", transcript)

async def main():
    scenarios = [
        ("Agent speaking - filler", "uh umm", 0.8, True),
        ("Agent speaking - command", "stop", 0.9, True),
        ("Agent silent - filler", "uh umm", 0.8, False),
        ("Agent speaking - mixed", "uh okay stop", 0.9, True),
    ]
    for label, text, conf, agent_speaking in scenarios:
        print("\n---", label, "---")
        result = await decide_and_handle(text, conf, agent_speaking, fake_interrupt_cb)
        print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
