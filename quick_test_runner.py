# quick_test_runner.py
# Standalone checks for interrupt_handler (bypasses repo pytest/conftest)
import asyncio
import sys
from extensions.interrupt_handler.handler import tokenize, is_filler_only, decide_and_handle

def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)

def ok(msg):
    print("PASS:", msg)

def run_sync_checks():
    # tokenize
    if tokenize("Uh, umm!") != ["uh", "umm"]:
        fail("tokenize failed")
    ok("tokenize")

    # is_filler_only cases
    if is_filler_only("uh umm", confidence=0.8) is not True:
        fail("is_filler_only should be True for filler with high conf")
    ok("is_filler_only filler-high-conf")

    if is_filler_only("uh stop", confidence=0.9) is not False:
        fail("is_filler_only should be False for filler+command")
    ok("is_filler_only filler+command")

    if is_filler_only("uh yes", confidence=0.9) is not False:
        fail("is_filler_only should be False for non-filler word")
    ok("is_filler_only non-filler word")

    if is_filler_only("", confidence=0.5) is not True:
        fail("is_filler_only should be True for empty transcript")
    ok("is_filler_only empty transcript")

async def run_async_checks():
    class MockAgent:
        def __init__(self):
            self.interrupted = False
        async def interrupt_cb(self, transcript, confidence):
            self.interrupted = True

    async def run_case(text, conf, agent_speaking):
        ag = MockAgent()
        res = await decide_and_handle(text, conf, agent_speaking, ag.interrupt_cb)
        return res, ag.interrupted

    res, interrupted = await run_case("uh umm", 0.8, True)
    if res["action"] != "ignored" or interrupted:
        fail("Agent speaking should IGNORE filler")
    ok("integration: agent speaking ignores filler")

    res, interrupted = await run_case("stop", 0.9, True)
    if res["action"] != "interrupt" or not interrupted:
        fail("Agent speaking should INTERRUPT on command")
    ok("integration: agent speaking interrupts on command")

    res, interrupted = await run_case("uh umm", 0.8, False)
    if res["action"] != "interrupt" or not interrupted:
        fail("Agent silent should count filler as user speech -> INTERRUPT")
    ok("integration: silent agent, filler -> interrupt")

def main():
    print("Running sync checks...")
    run_sync_checks()
    print("Running async integration checks...")
    asyncio.run(run_async_checks())
    print("ALL QUICK CHECKS PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()
