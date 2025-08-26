#!/usr/bin/env python3

"""
Test script to verify started_at timestamp functionality.
This script demonstrates the new timestamp features without requiring a full voice agent setup.
"""

import time
import sys
sys.path.insert(0, 'livekit-agents')

from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice.events import UserInputTranscribedEvent, SpeechCreatedEvent
from livekit.agents.voice.speech_handle import SpeechHandle


def test_chat_message_started_at():
    """Test ChatMessage with started_at field."""
    print("=== Testing ChatMessage started_at field ===")
    
    # Test message without started_at
    msg1 = ChatMessage(role="user", content=["Hello"])
    print(f"Message without started_at: {msg1.started_at}")
    assert msg1.started_at is None, "Default started_at should be None"
    
    # Test message with started_at
    speech_start_time = time.time()
    msg2 = ChatMessage(role="user", content=["Hello"], started_at=speech_start_time)
    print(f"Message with started_at: {msg2.started_at}")
    assert msg2.started_at == speech_start_time, "started_at should match the provided time"
    
    print("✓ ChatMessage started_at field works correctly\n")


def test_chat_context_timestamps():
    """Test ChatContext with relative timestamps."""
    print("=== Testing ChatContext relative timestamps ===")
    
    session_start = time.time()
    ctx = ChatContext()
    
    # Add messages with timestamps
    speech_start1 = session_start + 2.5  # User starts speaking 2.5s after session
    speech_start2 = session_start + 5.0  # Assistant starts speaking 5s after session
    
    msg1 = ctx.add_message(
        role="user", 
        content="What's the weather?",
        started_at=speech_start1
    )
    
    msg2 = ctx.add_message(
        role="assistant", 
        content="It's sunny today!",
        started_at=speech_start2  
    )
    
    # Test to_dict with relative timestamps
    result = ctx.to_dict(
        exclude_started_at=False,
        include_relative_timestamps=True,
        session_started_at=session_start
    )
    
    items = result['items']
    print(f"User message relative timestamps: started_at_relative={items[0].get('started_at_relative')}")
    print(f"Assistant message relative timestamps: started_at_relative={items[1].get('started_at_relative')}")
    
    # Verify relative timestamps
    assert abs(items[0]['started_at_relative'] - 2.5) < 0.01, "User message relative timestamp should be ~2.5s"
    assert abs(items[1]['started_at_relative'] - 5.0) < 0.01, "Assistant message relative timestamp should be ~5.0s"
    
    print("✓ Relative timestamps calculated correctly\n")


def test_events_started_at():
    """Test events with started_at fields."""
    print("=== Testing Events started_at fields ===")
    
    speech_start = time.time()
    
    # Test UserInputTranscribedEvent
    user_event = UserInputTranscribedEvent(
        transcript="Hello there",
        is_final=True,
        started_at=speech_start
    )
    print(f"User event started_at: {user_event.started_at}")
    assert user_event.started_at == speech_start, "User event started_at should match"
    
    # Test SpeechCreatedEvent  
    speech_handle = SpeechHandle.create()
    speech_event = SpeechCreatedEvent(
        user_initiated=True,
        source="generate_reply",
        speech_handle=speech_handle,
        started_at=speech_start
    )
    print(f"Speech event started_at: {speech_event.started_at}")
    assert speech_event.started_at == speech_start, "Speech event started_at should match"
    
    print("✓ Events started_at fields work correctly\n")


def main():
    """Run all tests."""
    print("Testing started_at timestamp functionality...")
    print("=" * 60)
    
    try:
        test_chat_message_started_at()
        test_chat_context_timestamps()  
        test_events_started_at()
        
        print("=" * 60)
        print("🎉 All tests passed! The started_at timestamp functionality is working correctly.")
        print("\nKey features implemented:")
        print("• ChatMessage.started_at field for tracking actual speech start times")
        print("• Events with started_at timestamps (UserInputTranscribedEvent, SpeechCreatedEvent)")
        print("• ChatContext.to_dict() with relative timestamp support")
        print("• Relative timestamps calculated from session start time")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()