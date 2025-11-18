"""
Comprehensive test suite for IntelligentInterruptionHandler
Run with: pytest test_interruption_handler.py -v
"""

import pytest
import asyncio
import time
from livekit.agents.interruption import (
    IntelligentInterruptionHandler,
    InterruptionType,
    LiveKitInterruptionWrapper
)


class TestBasicFillerDetection:
    """Test basic filler word detection"""
    
    @pytest.mark.asyncio
    async def test_single_filler_when_speaking(self):
        """Single filler should be ignored when agent is speaking"""
        handler = IntelligentInterruptionHandler(ignored_words=['uh', 'um', 'umm', 'hmm'])
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("umm", confidence=0.8)
        assert result == False, "Should not interrupt on single filler"
    
    @pytest.mark.asyncio
    async def test_multiple_fillers_when_speaking(self):
        """Multiple fillers should be ignored when agent is speaking"""
        handler = IntelligentInterruptionHandler(ignored_words=['uh', 'um', 'umm', 'hmm'])
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("uh umm hmm", confidence=0.8)
        assert result == False, "Should not interrupt on multiple fillers"
    
    @pytest.mark.asyncio
    async def test_filler_when_quiet(self):
        """Fillers should be registered as valid speech when agent is quiet"""
        handler = IntelligentInterruptionHandler(ignored_words=['uh', 'um', 'umm', 'hmm'])
        handler.set_agent_speaking(False)
        
        result = await handler.process_transcript("umm", confidence=0.8)
        assert result == False, "Should register but not interrupt when quiet"
        
        # Check classification
        history = handler.get_interruption_history(limit=1)
        assert history[0].classification == InterruptionType.VALID_SPEECH


class TestRealInterruptions:
    """Test detection of real user interruptions"""
    
    @pytest.mark.asyncio
    async def test_stop_command(self):
        """'Stop' should interrupt agent immediately"""
        handler = IntelligentInterruptionHandler()
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("stop", confidence=0.9)
        assert result == True, "Should interrupt on 'stop' command"
    
    @pytest.mark.asyncio
    async def test_wait_command(self):
        """'Wait' should interrupt agent immediately"""
        handler = IntelligentInterruptionHandler()
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("wait one second", confidence=0.9)
        assert result == True, "Should interrupt on 'wait' command"
    
    @pytest.mark.asyncio
    async def test_mixed_filler_and_command(self):
        """Mixed filler + command should still interrupt"""
        handler = IntelligentInterruptionHandler(ignored_words=['uh', 'um', 'okay'])
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("umm okay stop", confidence=0.8)
        assert result == True, "Should interrupt when command is present"
    
    @pytest.mark.asyncio
    async def test_sentence_with_filler(self):
        """Sentence containing filler should still interrupt"""
        handler = IntelligentInterruptionHandler(ignored_words=['uh'])
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("uh no not that one", confidence=0.85)
        assert result == True, "Should interrupt on meaningful content"


class TestConfidenceThreshold:
    """Test confidence-based filtering"""
    
    @pytest.mark.asyncio
    async def test_low_confidence_ignored(self):
        """Low confidence speech should be ignored when agent speaking"""
        handler = IntelligentInterruptionHandler(confidence_threshold=0.7)
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("hmm yeah", confidence=0.4)
        assert result == False, "Should ignore low confidence speech"
        
        history = handler.get_interruption_history(limit=1)
        assert history[0].classification == InterruptionType.LOW_CONFIDENCE
    
    @pytest.mark.asyncio
    async def test_high_confidence_processed(self):
        """High confidence speech should be processed normally"""
        handler = IntelligentInterruptionHandler(confidence_threshold=0.7)
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("stop please", confidence=0.9)
        assert result == True, "Should process high confidence speech"
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_exact(self):
        """Speech at exact threshold should be processed"""
        handler = IntelligentInterruptionHandler(confidence_threshold=0.7)
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("wait", confidence=0.7)
        assert result == True, "Should process speech at threshold"


class TestDynamicConfiguration:
    """Test runtime configuration updates"""
    
    @pytest.mark.asyncio
    async def test_add_ignored_words(self):
        """Should be able to add words to ignored list"""
        handler = IntelligentInterruptionHandler(
            ignored_words=['uh'],
            allow_runtime_updates=True
        )
        handler.set_agent_speaking(True)
        
        # Initially "okay" should interrupt
        result1 = await handler.process_transcript("okay", confidence=0.8)
        assert result1 == True, "Should interrupt before 'okay' is added"
        
        # Add "okay" to ignored list
        await handler.update_ignored_words(['okay'], append=True)
        
        # Now "okay" should be ignored
        result2 = await handler.process_transcript("okay", confidence=0.8)
        assert result2 == False, "Should ignore after 'okay' is added"
    
    @pytest.mark.asyncio
    async def test_replace_ignored_words(self):
        """Should be able to replace entire ignored list"""
        handler = IntelligentInterruptionHandler(
            ignored_words=['uh', 'um'],
            allow_runtime_updates=True
        )
        handler.set_agent_speaking(True)
        
        # Replace with new list
        await handler.update_ignored_words(['hmm', 'ah'], append=False)
        
        # Old words should now interrupt
        result1 = await handler.process_transcript("uh", confidence=0.8)
        assert result1 == True, "Should interrupt on previously ignored word"
        
        # New words should be ignored
        result2 = await handler.process_transcript("hmm", confidence=0.8)
        assert result2 == False, "Should ignore newly added word"
    
    @pytest.mark.asyncio
    async def test_updates_disabled(self):
        """Updates should be rejected when disabled"""
        handler = IntelligentInterruptionHandler(
            ignored_words=['uh'],
            allow_runtime_updates=False
        )
        
        # Attempt update (should be ignored)
        await handler.update_ignored_words(['okay'], append=True)
        
        # Original list should remain
        stats = handler.get_statistics()
        assert 'okay' not in stats['ignored_words']


class TestMultiLanguage:
    """Test multi-language filler support"""
    
    @pytest.mark.asyncio
    async def test_hindi_fillers(self):
        """Hindi fillers should be detected"""
        handler = IntelligentInterruptionHandler(
            ignored_words=['haan', 'han', 'achha']
        )
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("haan", confidence=0.8)
        assert result == False, "Should ignore Hindi filler"
    
    @pytest.mark.asyncio
    async def test_mixed_language_fillers(self):
        """Mixed language fillers should all be detected"""
        handler = IntelligentInterruptionHandler(
            ignored_words=['uh', 'um', 'haan', 'achha']
        )
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("uh haan", confidence=0.8)
        assert result == False, "Should ignore mixed language fillers"
    
    @pytest.mark.asyncio
    async def test_mixed_language_with_command(self):
        """Mixed language with command should interrupt"""
        handler = IntelligentInterruptionHandler(
            ignored_words=['haan', 'theek']
        )
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("haan theek stop karo", confidence=0.8)
        assert result == True, "Should interrupt on command in any language"


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.mark.asyncio
    async def test_empty_transcript(self):
        """Empty transcript should not interrupt"""
        handler = IntelligentInterruptionHandler()
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("", confidence=0.8)
        assert result == False, "Empty transcript should not interrupt"
    
    @pytest.mark.asyncio
    async def test_whitespace_only(self):
        """Whitespace-only transcript should not interrupt"""
        handler = IntelligentInterruptionHandler()
        handler.set_agent_speaking(True)
        
        result = await handler.process_transcript("   ", confidence=0.8)
        assert result == False, "Whitespace should not interrupt"
    
    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        """Filler detection should be case-insensitive"""
        handler = IntelligentInterruptionHandler(ignored_words=['uh', 'um'])
        handler.set_agent_speaking(True)
        
        result1 = await handler.process_transcript("UH", confidence=0.8)
        result2 = await handler.process_transcript("Um", confidence=0.8)
        result3 = await handler.process_transcript("uM", confidence=0.8)
        
        assert result1 == False, "Should ignore uppercase filler"
        assert result2 == False, "Should ignore mixed case filler"
        assert result3 == False, "Should ignore mixed case filler"
    
    @pytest.mark.asyncio
    async def test_rapid_state_changes(self):
        """Should handle rapid agent speaking state changes"""
        handler = IntelligentInterruptionHandler(ignored_words=['uh'])
        
        for i in range(100):
            handler.set_agent_speaking(i % 2 == 0)
            result = await handler.process_transcript(
                "uh" if i % 3 == 0 else "stop",
                confidence=0.8
            )
            # Verify no crashes or inconsistencies
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Should handle concurrent transcript processing"""
        handler = IntelligentInterruptionHandler()
        handler.set_agent_speaking(True)
        
        # Process multiple transcripts concurrently
        tasks = [
            handler.process_transcript(f"test {i}", confidence=0.8)
            for i in range(50)
        ]
        results = await asyncio.gather(*tasks)
        
        # All should complete without error
        assert len(results) == 50
        assert all(r is not None for r in results)


class TestStatistics:
    """Test statistics and logging functionality"""
    
    @pytest.mark.asyncio
    async def test_interruption_history(self):
        """Should track interruption history"""
        handler = IntelligentInterruptionHandler()
        handler.set_agent_speaking(True)
        
        await handler.process_transcript("umm", confidence=0.8)
        await handler.process_transcript("stop", confidence=0.9)
        
        history = handler.get_interruption_history()
        assert len(history) == 2, "Should have 2 events in history"
        assert history[0].text == "umm"
        assert history[1].text == "stop"
    
    @pytest.mark.asyncio
    async def test_limited_history(self):
        """Should return limited history when requested"""
        handler = IntelligentInterruptionHandler()
        handler.set_agent_speaking(True)
        
        for i in range(10):
            await handler.process_transcript(f"test {i}", confidence=0.8)
        
        history = handler.get_interruption_history(limit=3)
        assert len(history) == 3, "Should return only 3 most recent events"
    
    @pytest.mark.asyncio
    async def test_statistics_accuracy(self):
        """Statistics should accurately reflect events"""
        handler = IntelligentInterruptionHandler(ignored_words=['uh'])
        handler.set_agent_speaking(True)
        
        # 3 fillers, 2 real interruptions
        await handler.process_transcript("uh", confidence=0.8)
        await handler.process_transcript("uh", confidence=0.8)
        await handler.process_transcript("stop", confidence=0.9)
        await handler.process_transcript("wait", confidence=0.9)
        await handler.process_transcript("uh", confidence=0.8)
        
        stats = handler.get_statistics()
        assert stats['total_events'] == 5
        assert stats['by_type']['filler_only'] == 3
        assert stats['by_type']['real_interruption'] == 2


class TestIntegration:
    """Integration tests simulating real scenarios"""
    
    @pytest.mark.asyncio
    async def test_natural_conversation_flow(self):
        """Simulate a natural conversation with mixed inputs"""
        handler = IntelligentInterruptionHandler(
            ignored_words=['uh', 'um', 'umm', 'hmm'],
            confidence_threshold=0.6
        )
        
        # Agent starts speaking
        handler.set_agent_speaking(True)
        
        # User says fillers (should be ignored)
        r1 = await handler.process_transcript("hmm", confidence=0.7)
        assert r1 == False
        
        # User interrupts (should stop agent)
        r2 = await handler.process_transcript("wait a moment", confidence=0.85)
        assert r2 == True
        
        # Agent stops
        handler.set_agent_speaking(False)
        
        # User responds with filler (should be valid)
        r3 = await handler.process_transcript("uh", confidence=0.8)
        assert r3 == False
        
        # Check history
        history = handler.get_interruption_history()
        assert len(history) == 3
    
    @pytest.mark.asyncio
    async def test_background_noise_handling(self):
        """Simulate background noise with low confidence"""
        handler = IntelligentInterruptionHandler(confidence_threshold=0.7)
        handler.set_agent_speaking(True)
        
        # Background murmur (low confidence)
        r1 = await handler.process_transcript("hmm yeah", confidence=0.4)
        assert r1 == False, "Should ignore low confidence background noise"
        
        # Clear speech (high confidence)
        r2 = await handler.process_transcript("stop", confidence=0.9)
        assert r2 == True, "Should process clear speech"


# Performance benchmarks
class TestPerformance:
    """Performance and latency tests"""
    
    @pytest.mark.asyncio
    async def test_processing_latency(self):
        """Processing should be fast (<5ms per transcript)"""
        handler = IntelligentInterruptionHandler()
        handler.set_agent_speaking(True)
        
        start = time.time()
        for _ in range(100):
            await handler.process_transcript("test transcript", confidence=0.8)
        elapsed = time.time() - start
        
        avg_latency = elapsed / 100
        assert avg_latency < 0.005, f"Average latency too high: {avg_latency*1000:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Should not accumulate excessive memory"""
        handler = IntelligentInterruptionHandler()
        handler.set_agent_speaking(True)
        
        # Process 1000 transcripts
        for i in range(1000):
            await handler.process_transcript(f"test {i}", confidence=0.8)
        
        history = handler.get_interruption_history()
        assert len(history) == 1000, "Should store all events"
        # In production, consider implementing history rotation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
import time

class TestRobustness:
    """Enhanced robustness testing"""
    
    @pytest.mark.asyncio
    async def test_rapid_speech_burst(self):
        """Test burst of rapid inputs"""
        handler = IntelligentInterruptionHandler()
        handler.set_agent_speaking(True)
        
        # 50 rapid inputs
        tasks = [
            handler.process_transcript("test" if i%3 else "uh", 0.8)
            for i in range(50)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 50
    
    @pytest.mark.asyncio  
    async def test_stress_fast_switching(self):
        """Test rapid state switching under load"""
        handler = IntelligentInterruptionHandler()
        
        for i in range(100):
            handler.set_agent_speaking(i % 2 == 0)
            await handler.process_transcript("test", 0.8)
        
        assert handler.get_statistics()['total_events'] == 100
