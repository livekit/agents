#!/usr/bin/env python3

"""
Example usage of the new started_at timestamp functionality.
This demonstrates how the timestamps can be used for audio alignment and analytics.
"""

import time
import sys
sys.path.insert(0, 'livekit-agents')

from livekit.agents.llm import ChatContext


def simulate_voice_conversation():
    """Simulate a voice conversation with realistic timing."""
    print("=== Simulating Voice Conversation with Timestamps ===")
    
    # Session starts (would be set by AgentSession)
    session_start = time.time() 
    print(f"Session started at: {session_start}")
    
    # Create conversation context
    ctx = ChatContext()
    
    # Simulate conversation events
    events = [
        (2.1, "user", "Hi there, how are you?", "User starts speaking at 2.1s"),
        (5.8, "assistant", "I'm doing great, thank you for asking!", "Assistant starts speaking at 5.8s"),
        (10.2, "user", "What's the weather like today?", "User asks about weather at 10.2s"),
        (14.5, "assistant", "It's sunny and 72 degrees outside today.", "Assistant responds at 14.5s"),
    ]
    
    # Add messages to context
    for relative_time, role, content, description in events:
        print(f"  {description}")
        
        # Calculate absolute timestamp
        absolute_started_at = session_start + relative_time
        
        ctx.add_message(
            role=role,
            content=content,
            started_at=absolute_started_at
        )
    
    return ctx, session_start


def demonstrate_audio_alignment(ctx: ChatContext, session_start: float):
    """Show how timestamps can be used for audio alignment."""
    print(f"\n=== Audio Waveform Alignment Demo ===")
    
    # Get chat history with relative timestamps for audio alignment
    history = ctx.to_dict(
        exclude_timestamp=False,  # Keep created_at for comparison
        exclude_started_at=False,  # Keep started_at
        include_relative_timestamps=True,
        session_started_at=session_start
    )
    
    print("Chat items with relative timestamps for waveform alignment:")
    print("-" * 70)
    
    for i, item in enumerate(history['items'], 1):
        role = item['role']
        content = item['content'][0] if item['content'] else ""
        started_at_rel = item.get('started_at_relative')
        created_at_rel = item.get('created_at_relative')
        
        print(f"{i}. [{role.upper()}] \"{content}\"")
        if started_at_rel is not None:
            print(f"   🎤 Speech started at: {started_at_rel:.1f}s (waveform position)")
        if created_at_rel is not None:
            print(f"   📝 Message created at: {created_at_rel:.1f}s (processing time)")
        print()


def demonstrate_analytics(ctx: ChatContext):
    """Show how timestamps can be used for conversation analytics."""
    print("=== Conversation Analytics Demo ===")
    
    messages = []
    for item in ctx.items:
        if hasattr(item, 'role') and hasattr(item, 'started_at'):
            messages.append({
                'role': item.role,
                'started_at': item.started_at,
                'created_at': item.created_at,
                'content': item.text_content or ""
            })
    
    # Calculate metrics
    user_messages = [m for m in messages if m['role'] == 'user']
    assistant_messages = [m for m in messages if m['role'] == 'assistant']
    
    # Response times (time from user speech start to assistant speech start)
    response_times = []
    for i, user_msg in enumerate(user_messages):
        # Find the next assistant message
        for asst_msg in assistant_messages:
            if asst_msg['started_at'] and user_msg['started_at'] and asst_msg['started_at'] > user_msg['started_at']:
                response_time = asst_msg['started_at'] - user_msg['started_at']
                response_times.append(response_time)
                print(f"Response time #{i+1}: {response_time:.1f}s")
                break
    
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        print(f"\n📊 Average response time: {avg_response_time:.1f}s")
        print(f"📊 Fastest response: {min(response_times):.1f}s") 
        print(f"📊 Slowest response: {max(response_times):.1f}s")


def demonstrate_export_formats(ctx: ChatContext, session_start: float):
    """Show different export formats for various use cases."""
    print(f"\n=== Export Format Examples ===")
    
    # Format 1: For transcript (human readable)
    print("1. Transcript format (exclude internal timestamps):")
    transcript = ctx.to_dict(
        exclude_timestamp=True,
        exclude_started_at=True,
        exclude_function_call=True
    )
    for item in transcript['items']:
        role = item['role'].upper()
        content = item['content'][0] if item['content'] else ""
        print(f"   {role}: {content}")
    
    print("\n2. Analytics format (include all timestamps):")
    analytics = ctx.to_dict(
        exclude_timestamp=False,
        exclude_started_at=False, 
        include_relative_timestamps=True,
        session_started_at=session_start
    )
    print(f"   Items with timing data: {len(analytics['items'])}")
    
    print("\n3. Audio sync format (relative timestamps only):")
    audio_sync = ctx.to_dict(
        exclude_timestamp=True,  # Don't need processing times
        exclude_started_at=False,  # Keep speech start times
        include_relative_timestamps=True,  # For waveform alignment
        session_started_at=session_start
    )
    for item in audio_sync['items']:
        if 'started_at_relative' in item:
            role = item['role']
            rel_time = item['started_at_relative']
            print(f"   {rel_time:.1f}s: {role} speech starts")


def main():
    """Run the demonstration."""
    print("LiveKit Agents - started_at Timestamps Demo")
    print("=" * 60)
    
    # Simulate conversation
    ctx, session_start = simulate_voice_conversation()
    
    # Demonstrate various use cases
    demonstrate_audio_alignment(ctx, session_start)
    demonstrate_analytics(ctx)
    demonstrate_export_formats(ctx, session_start)
    
    print("=" * 60)
    print("✨ This demonstrates how started_at timestamps enable:")
    print("   • Precise audio waveform alignment (0:00 = session start)")
    print("   • Conversation analytics (response times, speech patterns)")
    print("   • Multiple export formats for different use cases")
    print("   • Distinction between speech start and processing times")


if __name__ == "__main__":
    main()