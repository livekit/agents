"""Test script for Azure TTS streaming implementation."""

import aiohttp
import asyncio
import os
import wave
from pathlib import Path

from livekit.plugins import azure


async def test_chunked_stream():
    """Test non-streaming synthesis (HTTP POST)."""
    print("\n=== Testing ChunkedStream (HTTP POST) ===")
    
    async with aiohttp.ClientSession() as session:
        tts = azure.TTS(
            speech_key=os.environ.get("SUBSCRIPTION_SPEECH_KEY"),
            speech_region=os.environ.get("SUBSCRIPTION_SPEECH_REGION", "eastus"),
            voice="en-US-Ava:DragonHDLatestNeural",
            sample_rate=24000,
            http_session=session,
        )
        
        test_text = "Hello! This is a test of Azure text to speech using HTTP mode."
        
        print(f"Synthesizing: {test_text}")
        stream = tts.synthesize(test_text)
        
        # Collect audio
        audio_data = bytearray()
        async for audio in stream:
            audio_data.extend(audio.frame.data)
        
        print(f"Generated {len(audio_data)} bytes of audio")
        
        # Save to file
        output_file = "test_chunked.wav"
        with wave.open(output_file, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_data)
        
        print(f"Saved to {output_file}")
        return audio_data


async def test_synthesize_stream():
    """Test streaming synthesis with TextStream (WebSocket v2)."""
    import time
    print("\n=== Testing SynthesizeStream (Azure SDK WebSocket v2) ===")
    
    async with aiohttp.ClientSession() as session:
        tts = azure.TTS(
            speech_key=os.environ.get("SUBSCRIPTION_SPEECH_KEY"),
            speech_region=os.environ.get("SUBSCRIPTION_SPEECH_REGION", "eastus"),
            voice="en-US-Ava:DragonHDLatestNeural",
            sample_rate=24000,
            http_session=session,
        )
        
        stream = tts.stream()
        
        # Track audio chunks as they arrive
        audio_data = bytearray()
        chunk_count = 0
        start_time = time.time()
        
        async def receive_audio():
            nonlocal chunk_count
            async for audio in stream:
                chunk_count += 1
                chunk_size = len(audio.frame.data)
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.2f}s] Audio chunk {chunk_count}: {chunk_size} bytes")
                audio_data.extend(audio.frame.data)
        
        # Start receiving audio
        receive_task = asyncio.create_task(receive_audio())
        
        # Stream text incrementally (simulating LLM output)
        text_chunks = [
            "First sentence here. This",
            "is a test ",
            "of Azure text to speech ",
            "using streaming mode. ",
            "Notice how audio arrives ",
            "as we send text chunks!"
        ]
        
        print("Streaming text chunks:")
        for i, chunk in enumerate(text_chunks, 1):
            elapsed = time.time() - start_time
            print(f"  [{elapsed:.2f}s] Sending chunk {i}: '{chunk.strip()}'")
            stream.push_text(chunk)
            await asyncio.sleep(0.2)  # Reduced delay to see interleaving better
        
        # Mark end of input and wait
        elapsed = time.time() - start_time
        print(f"  [{elapsed:.2f}s] Calling end_input()")
        stream.end_input()
        await receive_task
        await stream.aclose()
        
        print(f"\nGenerated {len(audio_data)} bytes in {chunk_count} chunks")
        
        # Save to file
        output_file = "test_streaming.wav"
        with wave.open(output_file, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_data)
        
        print(f"Saved to {output_file}")
        return audio_data


async def test_multiple_segments():
    """Test multiple segments with separate stream instances."""
    print("\n=== Testing Multiple Segments ===")
    
    async with aiohttp.ClientSession() as session:
        tts = azure.TTS(
            speech_key=os.environ.get("SUBSCRIPTION_SPEECH_KEY"),
            speech_region=os.environ.get("SUBSCRIPTION_SPEECH_REGION", "eastus"),
            voice="en-US-Ava:DragonHDLatestNeural",
            sample_rate=24000,
            http_session=session,
        )
        
        # Send multiple segments using separate stream instances
        sentences = [
            "First sentence here.",
            "Second sentence follows.",
            "Third and final sentence."
        ]
        
        all_audio_data = bytearray()
        
        print("Sending segments (separate streams):")
        for i, sentence in enumerate(sentences, 1):
            import time
            print(f"  Segment {i}: '{sentence}'")
            
            # Create a new stream for each segment (recommended approach)
            stream = tts.stream()
            
            audio_data = bytearray()
            first_audio_time = None
            
            async def receive_audio():
                nonlocal first_audio_time
                async for audio in stream:
                    if first_audio_time is None:
                        first_audio_time = time.time()
                    audio_data.extend(audio.frame.data)
            
            receive_task = asyncio.create_task(receive_audio())
            
            text_start_time = time.time()
            stream.push_text(sentence)
            stream.end_input()
            
            await receive_task
            await stream.aclose()
            
            latency = first_audio_time - text_start_time if first_audio_time else 0
            all_audio_data.extend(audio_data)
            print(f"    Generated {len(audio_data)} bytes, Latency: {latency:.2f}s")
        
        print(f"\nCompleted {len(sentences)} segments, {len(all_audio_data)} bytes total")


async def test_with_deployment_id():
    """Test with custom deployment ID (if configured)."""
    deployment_id = os.environ.get("AZURE_DEPLOYMENT_ID")
    if not deployment_id:
        print("\n=== Skipping deployment test (AZURE_DEPLOYMENT_ID not set) ===")
        return
    
    print(f"\n=== Testing with Deployment ID: {deployment_id} ===")
    
    tts = azure.TTS(
        speech_key=os.environ.get("SUBSCRIPTION_SPEECH_KEY"),
        speech_region=os.environ.get("SUBSCRIPTION_SPEECH_REGION", "eastus"),
        voice="YourCustomVoiceName",  # Replace with your custom voice
        deployment_id=deployment_id,
        sample_rate=16000,
    )
    
    stream = tts.stream()
    
    audio_data = bytearray()
    async for event in stream.synthesize("Testing custom deployment."):
        if event.type == "audio":
            audio_data.extend(event.frame.data)
    
    print(f"Generated {len(audio_data)} bytes with custom deployment")


async def main():
    """Run all tests."""
    # Check environment
    if not os.environ.get("SUBSCRIPTION_SPEECH_KEY"):
        print("ERROR: SUBSCRIPTION_SPEECH_KEY environment variable not set")
        print("\nSet it with:")
        print("  $env:SUBSCRIPTION_SPEECH_KEY='your-key-here'  # PowerShell")
        print("  export SUBSCRIPTION_SPEECH_KEY='your-key-here'  # Bash")
        return
    
    if not os.environ.get("SUBSCRIPTION_SPEECH_REGION"):
        print("WARNING: SUBSCRIPTION_SPEECH_REGION not set, using default 'eastus'")
    
    print("Azure TTS Streaming Test")
    print("=" * 50)
    
    try:
        # Test 1: HTTP (non-streaming)
        await test_chunked_stream()
        
        # Test 2: WebSocket streaming
        await test_synthesize_stream()
        
        # Test 3: Multiple segments
        await test_multiple_segments()
        
        # Test 4: Custom deployment (optional)
        await test_with_deployment_id()
        
        print("\n" + "=" * 50)
        print("[SUCCESS] All tests completed successfully!")
        print("\nCheck the generated WAV files:")
        print("  - test_chunked.wav (HTTP mode)")
        print("  - test_streaming.wav (Streaming mode)")
        
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
