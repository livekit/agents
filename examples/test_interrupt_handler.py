import asyncio
from agents.extensions.interrupt_handler.handler import InterruptHandler
import asyncio
from agents.extensions.interrupt_handler.handler import InterruptHandler

# Simulated speaking state
agent_speaking = True



# Simulated speaking state
agent_speaking = True

def is_agent_speaking():
    return agent_speaking

def stop_agent():
    global agent_speaking
    print(">>> Agent interrupted and stopped speaking")
    agent_speaking = False


interrupt_handler = InterruptHandler(
    is_agent_speaking=is_agent_speaking,
    stop_agent=stop_agent,
    logger=print,
)

async def simulate_input(text, confidence=1.0):
    print(f"\nUser says: {text}")
    await interrupt_handler.handle_asr(text, confidence)


async def run_tests():
    print("=== Starting Interrupt Handler Test ===")

    global agent_speaking
    agent_speaking = True

    # Agent is speaking
    await simulate_input("umm")          # ignored filler
    await simulate_input("haan")         # ignored filler
    await simulate_input("wait stop")    # valid interruption

    # Agent is now silent
    await simulate_input("umm")          # user speech (not ignored)

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(run_tests())
