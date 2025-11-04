import asyncio
import logging
import aiohttp
from pathlib import Path
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents.inference import TTS as InferenceTTS  # Fixed import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts-test")

load_dotenv()

async def test_apply_text_normalization():
    """Test apply_text_normalization through the gateway by synthesizing the same text with different settings"""
    
    # Test phrase that benefits from normalization
    test_phrase = "I have 42 apples, $1,234.56, and Dr. Smith's phone is 555-123-4567 on 12/25/2024 Visit https://example.com for NASA information"
    
    # Settings to test
    settings_cycle = [
        {"apply_text_normalization": "off"},
        {"apply_text_normalization": "on"},
        {"apply_text_normalization": "auto"},
    ]
    
    # Create output directory
    output_dir = Path("tts_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create our own HTTP session for standalone use
    async with aiohttp.ClientSession() as session:
        for setting_idx, settings in enumerate(settings_cycle, 1):
            logger.info(f"\n=== Setting {setting_idx}: {settings} ===\n")
            
            # Create a new InferenceTTS instance that connects through the gateway
            tts = InferenceTTS(
                model="elevenlabs/eleven_flash_v2",
                voice="bIHbv24MWmeRgasZH58o",
                extra_kwargs=settings,  # Pass extra settings here
                http_session=session,
            )
            
            try:
                # Use stream() instead of synthesize()
                stream = tts.stream()
                
                # Push text synchronously (not async!)
                logger.info(f"Synthesizing: '{test_phrase}'")
                stream.push_text(test_phrase)
                stream.flush()
                stream.end_input()
                
                # Collect all audio frames by iterating over the stream
                audio_frames = []
                async for ev in stream:
                    audio_frames.append(ev.frame)
                    logger.debug(f"Got audio frame: {len(ev.frame.data)} bytes")
                
                # Combine frames and save to WAV file
                if audio_frames:
                    combined_audio = rtc.combine_audio_frames(audio_frames)
                    wav_bytes = combined_audio.to_wav_bytes()
                    
                    # Save to file
                    setting_name = settings["apply_text_normalization"]
                    output_file = output_dir / f"normalization_gateway_{setting_name}.wav"
                    output_file.write_bytes(wav_bytes)
                    logger.info(f"Saved audio to: {output_file}")
                
            finally:
                await stream.aclose()
                await tts.aclose()
            
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(test_apply_text_normalization())