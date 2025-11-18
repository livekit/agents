"""End-to-end test of interruption handler with LiveKit agent"""

import asyncio
import logging
from livekit.agents.interruption import IntelligentInterruptionHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockSTTEvent:
    """Mock STT event for testing"""
    def __init__(self, text, confidence, is_final=True):
        self.alternatives = [type('obj', (), {
            'text': text,
            'confidence': confidence
        })]
        self.is_final = is_final

class MockAgent:
    """Mock agent to test integration"""
    def __init__(self):
        self.handler = IntelligentInterruptionHandler(
            ignored_words=['uh', 'um', 'umm', 'hmm', 'haan'],
            confidence_threshold=0.6,
            log_all_events=True
        )
        self.is_speaking = False
        self.interrupted_count = 0
        
    async def start_speaking(self):
        self.is_speaking = True
        self.handler.set_agent_speaking(True)
        logger.info("🎙️  Agent started speaking...")
        
    async def stop_speaking(self):
        self.is_speaking = False
        self.handler.set_agent_speaking(False)
        logger.info("🔇 Agent stopped speaking")
        
    async def on_interrupt(self):
        self.interrupted_count += 1
        logger.info(f"🛑 Agent interrupted! (Total: {self.interrupted_count})")
        await self.stop_speaking()
        
    async def handle_stt_event(self, event: MockSTTEvent):
        text = event.alternatives[0].text
        confidence = event.alternatives[0].confidence
        
        should_interrupt = await self.handler.process_transcript(
            text=text,
            confidence=confidence,
            is_final=event.is_final
        )
        
        if should_interrupt and self.is_speaking:
            await self.on_interrupt()
        
        return should_interrupt

async def test_end_to_end_conversation():
    """Simulate a complete conversation with interruptions"""
    
    print("\n" + "="*60)
    print("🧪 END-TO-END AGENT INTEGRATION TEST")
    print("="*60 + "\n")
    
    agent = MockAgent()
    
    # Scenario 1: Agent speaks, user says filler
    print("📝 Scenario 1: Agent speaking + user filler")
    await agent.start_speaking()
    event1 = MockSTTEvent("umm", 0.8)
    result1 = await agent.handle_stt_event(event1)
    assert result1 == False
    assert agent.is_speaking == True
    print("   ✅ Agent ignored filler and continued speaking\n")
    
    # Scenario 2: Agent speaks, user interrupts
    print("📝 Scenario 2: Agent speaking + real interruption")
    event2 = MockSTTEvent("wait a second", 0.9)
    result2 = await agent.handle_stt_event(event2)
    assert result2 == True
    assert agent.is_speaking == False
    print("   ✅ Agent stopped immediately on real interruption\n")
    
    # Scenario 3: Agent quiet, user says filler
    print("📝 Scenario 3: Agent quiet + user filler")
    event3 = MockSTTEvent("hmm", 0.8)
    result3 = await agent.handle_stt_event(event3)
    assert result3 == False
    print("   ✅ Filler registered as valid speech when agent quiet\n")
    
    # Scenario 4: Mixed filler and command
    print("📝 Scenario 4: Agent speaking + mixed filler + command")
    await agent.start_speaking()
    event4 = MockSTTEvent("umm okay stop", 0.85)
    result4 = await agent.handle_stt_event(event4)
    assert result4 == True
    print("   ✅ Agent detected command despite filler\n")
    
    # Scenario 5: Low confidence background noise
    print("📝 Scenario 5: Agent speaking + low confidence noise")
    await agent.start_speaking()
    event5 = MockSTTEvent("hmm yeah", 0.4)
    result5 = await agent.handle_stt_event(event5)
    assert result5 == False
    assert agent.is_speaking == True
    print("   ✅ Agent ignored low confidence background noise\n")
    
    # Print statistics
    stats = agent.handler.get_statistics()
    print("="*60)
    print("📊 FINAL STATISTICS")
    print("="*60)
    print(f"Total Events Processed: {stats['total_events']}")
    print(f"Events by Type: {stats['by_type']}")
    print(f"Total Interruptions: {agent.interrupted_count}")
    print(f"Ignored Words: {len(stats['ignored_words'])} configured")
    
    print("\n" + "="*60)
    print("✅ ALL END-TO-END TESTS PASSED")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_end_to_end_conversation())
