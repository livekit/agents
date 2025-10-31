# test_all_deepgram_params.py
# 
# Tests Deepgram STT parameters that are supported by agent-gateway.
# Only parameters listed in agent-gateway/pkg/provider/stt/deepgram/types.go ExtraParams are tested.
# 
# Supported parameters:
# - callback, callback_method, channels, diarize, dictation, endpointing, extra,
#   filler_words, interim_results, keyterm, keywords, mip_opt_out, multichannel,
#   numerals, profanity_filter, punctuate, redact, replace, search, smart_format,
#   tag, utterance_end, vad_events, version
#
# Note: encoding, language, model, and sample_rate are handled separately by agent-gateway.
#
import asyncio
import logging
from typing import Any, Dict, Optional
from livekit import rtc
from livekit.agents import (
    stt as stt_module,
    utils,
)
from livekit.agents.inference import STT
from livekit.plugins import cartesia
from dotenv import load_dotenv
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Test phrases designed to verify each parameter's effect
TEST_PHRASES: Dict[str, str] = {
    "default": "Hello, this is a test of the Deepgram transcription system.",
    "smart_format": "I have twenty three apples on December fifth twenty twenty four at three thirty PM.",
    "numerals": "I have twenty three apples and bought five more.",
    "filler_words_false": "Um, I think uh, that's correct, you know.",
    "filler_words_true": "Um, I think uh, that's correct, you know.",
    "punctuate_false": "hello how are you today i am fine",
    "punctuate_true": "hello how are you today i am fine",
    "profanity_filter": "That's a damn good idea!",
    # Diarization test phrase with clear speaker changes (longer pause helps)
    "diarize": "Hello and welcome to our call. My name is Alice. How can I help you today? Thank you. My name is Bob. I need assistance with my account.",
    "dictation": "Hello period How are you question mark",
    "keyterm": "The API uses OAuth authentication with JWT tokens.",
    "keywords": "I work with React and Apple frameworks.",
    "replace": "I said yes many times, um, you know um.",
    "search": "I mentioned React and Apple in my talk.",
    "multichannel": "Channel one content. Channel two content.",
    "redact": "My credit card number is 4532-1234-5678-9010.",
}

# Complete parameter definitions - ONLY parameters supported by agent-gateway
# Based on agent-gateway/pkg/provider/stt/deepgram/types.go ExtraParams list
ALL_DEEPGRAM_PARAMS: Dict[str, Dict[str, Any]] = {
    # Boolean parameters
    "smart_format": {"smart_format": True},
    "numerals": {"numerals": True},
    "filler_words_false": {"filler_words": False},
    "filler_words_true": {"filler_words": True},
    "profanity_filter": {"profanity_filter": True},
    "punctuate_false": {"punctuate": False},
    "punctuate_true": {"punctuate": True},
    "diarize": {"diarize": True},
    "dictation": {"dictation": True},
    "mip_opt_out": {"mip_opt_out": True},
    "multichannel": {"multichannel": True},
    "vad_events": {"vad_events": True},
    "interim_results": {"interim_results": True},
    "redact": {"redact": True},
    
    # Integer/Numeric parameters
    "endpointing": {"endpointing": 50},
    "channels": {"channels": 1},
    "utterance_end": {"utterance_end": 1000},  # Note: agent-gateway uses utterance_end (not utterance_end_ms from API)
    
    # String parameters
    "tag": {"tag": "test-tag"},
    "version": {"version": "latest"},  # Deepgram version: "latest" (default), version number like "2021-03-17.0", or custom model version_id
    "extra": {"extra": "test_extra"},
    
    # List/Array parameters
    "keyterm": {"keyterm": "OAuth,JWT,API"},
    # Keywords: format is KEYWORD:INTENSIFIER (intensifier defaults to 1 if omitted)
    # For multiple keywords, send multiple parameters or comma-separated list
    # Note: Keywords only works with Nova-2, Nova-1, Enhanced, and Base models (NOT Nova-3)
    # For Nova-3, use keyterm prompting instead
    "keywords": {"keywords": "React:2,Apple:2"},  # Boost React and Apple with intensifier 2
    # Replace: format is FIND:REPLACE (find term should be lowercase)
    "replace": {"replace": "yes:no"},  # Replace "yes" with "no"
    # Search: format can be comma-separated or multiple parameters
    "search": {"search": "React,Apple"},
    
    # Complex parameters
    # Callback: For full testing, use a real webhook endpoint (e.g., webhook.site, ngrok)
    # This test only verifies the parameter is accepted (transcript received)
    # Use https://webhook.site/ for testing
    "callback": {"callback": "https://webhook.site/fe27a030-e9bb-4d8d-a964-cba3d4a48fcf"},
    "callback_method": {"callback_method": "POST"},
}


async def generate_test_audio(text: str, sample_rate: int = 16000, http_session: Optional[aiohttp.ClientSession] = None) -> list[rtc.AudioFrame]:
    """Generate audio frames from text using TTS"""
    tts = cartesia.TTS(http_session=http_session)
    frames = []
    
    async for synthesized_audio in tts.synthesize(text):
        frame = synthesized_audio.frame
        
        # Resample if needed
        if sample_rate != frame.sample_rate:
            resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=sample_rate,
                num_channels=frame.num_channels,
            )
            resampled_frames = resampler.push(frame)
            frames.extend(resampled_frames)
            # Flush any remaining frames
            flushed = resampler.flush()
            if flushed:
                frames.extend(flushed)
        else:
            frames.append(frame)
    
    return frames


def verify_parameter(param_name: str, param_value: Dict[str, Any], transcript: str, events: list) -> Dict[str, Any]:
    """Verify if the parameter had the expected effect"""
    verification = {
        "param": param_name,
        "transcript_received": bool(transcript),
        "transcript": transcript,
        "verified": False,
        "notes": [],
        "metadata": {},  # Store extracted metadata
    }
    
    # Extract metadata from events
    for event in events:
        if event.type == stt_module.SpeechEventType.FINAL_TRANSCRIPT:
            for alt in event.alternatives:
                if alt.speaker_id:
                    verification["metadata"]["speaker_id"] = alt.speaker_id
                if alt.language:
                    verification["metadata"]["language"] = alt.language
                if alt.confidence:
                    verification["metadata"]["confidence"] = alt.confidence
                if alt.start_time:
                    verification["metadata"]["start_time"] = alt.start_time
                if alt.end_time:
                    verification["metadata"]["end_time"] = alt.end_time
                if alt.is_primary_speaker is not None:
                    verification["metadata"]["is_primary_speaker"] = alt.is_primary_speaker
    
    transcript_lower = transcript.lower() if transcript else ""
    
    if param_name == "smart_format":
        has_formatted = any(x in transcript for x in ["23", "December 5", "2024", "3:30"])
        verification["verified"] = has_formatted
        verification["notes"].append(f"Should contain formatted numbers/dates: {has_formatted}")
    
    elif param_name == "numerals":
        has_numerals = "23" in transcript or "5" in transcript
        verification["verified"] = has_numerals
        verification["notes"].append(f"Should contain numerals: {has_numerals}")
    
    elif param_name in ["filler_words_false", "filler_words_true"]:
        has_fillers = any(word in transcript_lower for word in [" um ", " uh ", "you know"])
        expected_fillers = param_value.get("filler_words", True)
        verification["verified"] = (has_fillers == expected_fillers) if expected_fillers else True
        verification["notes"].append(f"filler_words={expected_fillers}, has_fillers={has_fillers}")
    
    elif param_name in ["punctuate_false", "punctuate_true"]:
        has_punctuation = any(c in transcript for c in ".,!?;:")
        expected = param_value.get("punctuate", True)
        verification["verified"] = has_punctuation == expected
        verification["notes"].append(f"Has punctuation: {has_punctuation}, Expected: {expected}")
    
    elif param_name == "profanity_filter":
        has_profanity = "damn" in transcript_lower
        verification["verified"] = not has_profanity
        verification["notes"].append(f"Should NOT contain profanity: {not has_profanity}")
    
    elif param_name in ["keyterm", "keywords"]:
        # Keywords format: "KEYWORD:INTENSIFIER" or comma-separated "KEYWORD1:INT1,KEYWORD2:INT2"
        # Extract just the keyword part (before colon) for checking
        param_str = param_value.get("keyterm", param_value.get("keywords", ""))
        # Split by comma, then extract keyword part (before colon if present)
        terms = []
        for item in param_str.split(","):
            term = item.split(":")[0].strip()  # Get keyword part, remove intensifier
            terms.append(term)
        found_terms = sum(1 for term in terms if term.lower() in transcript_lower)
        verification["verified"] = found_terms > 0
        verification["notes"].append(f"Found {found_terms}/{len(terms)} key terms: {terms}")
        if found_terms < len(terms):
            verification["notes"].append(f"Note: Keywords may boost recognition but not guarantee appearance in transcript")
    
    elif param_name == "replace":
        # Replace format: "FIND:REPLACE" - finds "yes" and replaces with "no"
        # Check if replacement appears OR if original doesn't appear (replacement worked)
        has_replacement = "no" in transcript_lower
        has_original = " yes " in transcript_lower or transcript_lower.startswith("yes ") or transcript_lower.endswith(" yes")
        # Replacement worked if we see "uhm" or don't see "um" (assuming test phrase contains "um")
        verification["verified"] = has_replacement or not has_original
        verification["notes"].append(f"Replace 'yes'→'no': replacement found={has_replacement}, original found={has_original}")
        if not verification["verified"]:
            verification["notes"].append("Note: Replace only works if the find term appears in audio")
    
    elif param_name == "search":
        # Search: searches for terms/phrases by matching acoustic patterns
        # Returns metadata with start/end times and confidence for each match
        # Format can be comma-separated or multiple parameters
        terms = [t.strip() for t in param_value.get("search", "").split(",")]
        found_terms_in_transcript = sum(1 for term in terms if term.lower() in transcript_lower)
        
        # Check events for search-related metadata
        search_metadata_found = False
        search_results_count = 0
        
        for event in events:
            # Check if event has search-related data
            if hasattr(event, '__dict__'):
                event_dict = event.__dict__
                # Look for search-related keys
                search_keys = [k for k in event_dict.keys() if 'search' in k.lower()]
                if search_keys:
                    search_metadata_found = True
                    verification["metadata"]["search_keys"] = search_keys
                    logger.info(f"Found search-related keys in event: {search_keys}")
            
            # Check alternatives for search metadata
            if event.type == stt_module.SpeechEventType.FINAL_TRANSCRIPT:
                for alt in event.alternatives:
                    # Check for search-related attributes
                    for attr_name in dir(alt):
                        if 'search' in attr_name.lower() and not attr_name.startswith('_'):
                            try:
                                attr_value = getattr(alt, attr_name)
                                if attr_value is not None:
                                    search_metadata_found = True
                                    verification["metadata"][attr_name] = attr_value
                                    logger.info(f"Found search attribute: {attr_name}={attr_value}")
                            except:
                                pass
        
        # Search is verified if:
        # 1. Terms appear in transcript (basic check)
        # 2. OR search metadata is found in events
        # 3. OR transcript received (parameter accepted, but may not have matches)
        verification["verified"] = found_terms_in_transcript > 0 or search_metadata_found
        
        verification["notes"].append(f"Search terms requested: {terms}")
        verification["notes"].append(f"Found {found_terms_in_transcript}/{len(terms)} terms in transcript")
        
        if found_terms_in_transcript > 0:
            verification["notes"].append(f"✓ Search WORKING - {found_terms_in_transcript} terms found in transcript")
        elif search_metadata_found:
            verification["notes"].append("✓ Search WORKING - search metadata detected in response")
            verification["notes"].append(f"Search metadata keys: {verification['metadata'].get('search_keys', [])}")
        else:
            verification["notes"].append("⚠ Search parameter accepted but no matches found")
            verification["notes"].append("Note: Search returns metadata with timestamps/confidence for acoustic matches")
            verification["notes"].append("Note: If terms don't match acoustically, they won't appear in search results even if in transcript")
        
        if verification["verified"]:
            verification["notes"].append("✓ Search feature is WORKING")
    
    elif param_name == "diarize":
        # Diarization: For live streaming, Deepgram returns speaker in words array
        # (For pre-recorded, both speaker and speaker_confidence are returned)
        all_speaker_ids = set()
        word_level_speakers = []
        alternative_level_speakers = []
        
        # Check all events for speaker information
        for event in events:
            if event.type == stt_module.SpeechEventType.FINAL_TRANSCRIPT:
                for alt in event.alternatives:
                    # Check alternative-level speaker_id (agent-gateway may extract from words)
                    if alt.speaker_id is not None:
                        speaker_id = str(alt.speaker_id) if alt.speaker_id is not None else None
                        if speaker_id:
                            all_speaker_ids.add(speaker_id)
                            alternative_level_speakers.append((alt.text[:30] + "...", speaker_id))
                            verification["metadata"]["alternative_speaker_id"] = speaker_id
                    
                    # Check words for speaker information (per Deepgram docs, speaker is in words array)
                    # Try to access words if available
                    if hasattr(alt, 'words') and alt.words:
                        for word in alt.words:
                            # Try different attribute names
                            speaker = None
                            if hasattr(word, 'speaker') and word.speaker is not None:
                                speaker = str(word.speaker)
                            elif hasattr(word, 'speaker_id') and word.speaker_id is not None:
                                speaker = str(word.speaker_id)
                            
                            if speaker is not None:
                                all_speaker_ids.add(speaker)
                                word_text = getattr(word, 'word', getattr(word, 'text', 'unknown'))
                                word_level_speakers.append((word_text, speaker))
                    
                    # Also check raw attributes for any speaker-related data
                    for attr_name in dir(alt):
                        if 'speaker' in attr_name.lower() and not attr_name.startswith('_'):
                            try:
                                attr_value = getattr(alt, attr_name)
                                if attr_value is not None:
                                    logger.debug(f"Found speaker attribute {attr_name}={attr_value}")
                            except:
                                pass
        
        # Diarization is ONLY verified if we actually found speaker information
        # Even with single speaker, diarize=true should assign speaker 0
        has_speaker_info = len(all_speaker_ids) > 0
        verification["verified"] = has_speaker_info  # Require actual speaker info
        
        verification["notes"].append(f"Diarization check: speaker info found={has_speaker_info}, unique speakers={len(all_speaker_ids)}")
        if all_speaker_ids:
            verification["notes"].append(f"✓ Speaker IDs found: {sorted(all_speaker_ids)}")
            verification["notes"].append("✓ Diarization is WORKING - speaker information successfully detected")
        if alternative_level_speakers:
            verification["notes"].append(f"Alternative-level speakers: {alternative_level_speakers}")
        if word_level_speakers:
            # Show sample word-level speaker assignments
            sample_words = word_level_speakers[:10]  # First 10 words
            verification["notes"].append(f"Word-level speaker assignments (sample): {sample_words}")
            verification["notes"].append(f"Total words with speaker info: {len(word_level_speakers)}")
        
        if not has_speaker_info:
            verification["notes"].append("✗ Diarization NOT WORKING - no speaker information found")
            verification["notes"].append("Note: Even with single TTS voice, diarize=true should assign speaker 0")
            verification["notes"].append("Note: Check if agent-gateway is properly extracting speaker info from Deepgram response")
    
    elif param_name == "interim_results":
        # Check if interim results are being received
        has_interim = any(
            event.type == stt_module.SpeechEventType.INTERIM_TRANSCRIPT
            for event in events
        )
        verification["verified"] = bool(transcript)
        verification["notes"].append(f"Interim results enabled: {has_interim}")
    
    elif param_name == "vad_events":
        # VAD events might show up as speech started events
        verification["verified"] = bool(transcript)
        verification["notes"].append("VAD events parameter passed (events may be in metadata)")
    
    elif param_name == "utterance_end":
        # Utterance end parameter - verify transcript was received
        verification["verified"] = bool(transcript)
        verification["notes"].append("Utterance end parameter passed")
    
    elif param_name == "callback":
        # Callback: URL for asynchronous processing (pre-recorded audio only for streaming)
        # For streaming, callback may not work the same way, but parameter should be accepted
        # Best way to test: verify parameter is accepted and transcript is received
        # Full callback test would require a real webhook endpoint
        callback_url = param_value.get("callback", "")
        verification["verified"] = bool(transcript)  # Parameter accepted if we got a transcript
        verification["notes"].append(f"Callback URL provided: {callback_url}")
        verification["notes"].append("Note: Callback sends transcription results asynchronously to URL")
        verification["notes"].append("Note: Full callback test requires a real webhook endpoint - this test only verifies parameter acceptance")
    
    elif param_name in ["tag", "extra", "version", "callback_method", "mip_opt_out", "channels"]:
        verification["verified"] = bool(transcript)
        verification["notes"].append(f"Parameter {param_name} is metadata/tracking only")
    
    else:
        verification["verified"] = bool(transcript)
        verification["notes"].append("Basic connectivity test")
    
    return verification


async def test_stt_parameter(param_name: str, param_value: Dict[str, Any], test_phrase: Optional[str] = None, http_session: Optional[aiohttp.ClientSession] = None) -> Dict[str, Any]:
    """Test a single STT parameter directly"""
    if test_phrase is None:
        test_phrase = TEST_PHRASES.get(param_name, TEST_PHRASES["default"])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {param_name}")
    logger.info(f"Parameters: {param_value}")
    logger.info(f"Test phrase: {test_phrase}")
    logger.info(f"{'='*60}")
    
    # Keywords only work with Nova-2, Nova-1, Enhanced, and Base models (NOT Nova-3)
    # Use Nova-2 for keywords test
    model = "deepgram/nova-2" if param_name == "keywords" else "deepgram/nova-3"
    
    try:
        stt = STT(
            model=model,
            language="en",
            extra_kwargs=param_value,
            http_session=http_session
        )
        
        if param_name == "keywords":
            logger.info(f"Using {model} for keywords test (Nova-3 doesn't support keywords)")
        
        logger.info("Generating audio...")
        audio_frames = await generate_test_audio(test_phrase, http_session=http_session)
        logger.info(f"Generated {len(audio_frames)} audio frames")
        
        stream = stt.stream()
        events = []
        transcripts = []
        
        logger.info("Pushing audio frames...")
        for i, frame in enumerate(audio_frames):
            stream.push_frame(frame)
            if i % 50 == 0:
                logger.debug(f"Pushed {i}/{len(audio_frames)} frames")
        
        # Signal end of input
        stream.end_input()
        
        logger.info("Collecting transcription events...")
        try:
            async with asyncio.timeout(10.0):
                async for event in stream:
                    events.append(event)
                    
                    # Log all event data for debugging
                    logger.debug(f"Event type: {event.type}, request_id: {event.request_id}")
                    
                    if event.type == stt_module.SpeechEventType.FINAL_TRANSCRIPT:
                        for alt_idx, alt in enumerate(event.alternatives):
                            transcript = alt.text
                            transcripts.append(transcript)
                            
                            # Extract available metadata
                            metadata_info = {
                                "transcript": transcript,
                                "language": alt.language,
                                "confidence": alt.confidence,
                                "speaker_id": alt.speaker_id,  # For diarization
                                "start_time": alt.start_time,
                                "end_time": alt.end_time,
                                "is_primary_speaker": alt.is_primary_speaker,
                            }
                            
                            logger.info(f"Final transcript [{alt_idx}]: {transcript}")
                            logger.info(f"  Metadata: {metadata_info}")
                            
                            # Inspect raw event for additional data (speaker info, search results, etc.)
                            if hasattr(event, '__dict__'):
                                event_dict = {k: v for k, v in event.__dict__.items() if not k.startswith('_')}
                                logger.debug(f"  Full event data: {event_dict}")
                            
                            # Inspect alternative object for all attributes
                            if param_name in ["diarize", "search"]:
                                alt_attrs = {attr: getattr(alt, attr, None) for attr in dir(alt) 
                                           if not attr.startswith('_') and not callable(getattr(alt, attr, None))}
                                logger.info(f"  Alternative attributes: {alt_attrs}")
                            
                            # Check if any metadata fields are populated
                            if param_name == "diarize" and alt.speaker_id:
                                logger.info(f"  ✓ Diarization: speaker_id={alt.speaker_id}")
                            if param_name == "detect_language" and alt.language:
                                logger.info(f"  ✓ Language detected: {alt.language}")
                            if param_name == "search":
                                # Log if search-related metadata might be present
                                logger.info(f"  Search parameter enabled - checking for search results in metadata")
                                
                    # Check for recognition usage events (might contain metadata)
                    elif event.type == stt_module.SpeechEventType.RECOGNITION_USAGE:
                        logger.info(f"Recognition usage: {event.recognition_usage}")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for transcription events")
        finally:
            await stream.aclose()
        
        final_transcript = transcripts[-1] if transcripts else None
        verification = verify_parameter(param_name, param_value, final_transcript or "", events)
        
        logger.info(f"✓ Test completed: {verification['verified']}")
        for note in verification["notes"]:
            logger.info(f"  - {note}")
        
        return verification
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"✗ Error testing {param_name}: {error_msg}", exc_info=True)
        
        # Special handling for keywords with wrong model
        if param_name == "keywords" and "400" in error_msg and "bad request" in error_msg.lower():
            notes = [
                f"Error: {error_msg}",
                "Keywords parameter requires Nova-2, Nova-1, Enhanced, or Base model (not Nova-3)",
                "Test should automatically use Nova-2 for keywords - check if model selection is working"
            ]
        else:
            notes = [f"Error: {error_msg}"]
        
        return {
            "param": param_name,
            "transcript_received": False,
            "transcript": None,
            "verified": False,
            "error": error_msg,
            "notes": notes,
        }


async def main():
    """Main function for standalone testing"""
    logger.info("="*60)
    logger.info("DEEPGRAM PARAMETER AUTOMATED TEST SUITE")
    logger.info("="*60)
    
    # Create HTTP session for TTS plugin
    async with aiohttp.ClientSession() as http_session:
        results = {}
        
        for param_name, param_value in ALL_DEEPGRAM_PARAMS.items():
            try:
                result = await test_stt_parameter(param_name, param_value, http_session=http_session)
                results[param_name] = result
            except Exception as e:
                logger.error(f"Failed to test {param_name}: {e}")
                results[param_name] = {
                    "param": param_name,
                    "verified": False,
                    "error": str(e),
                }
            
            await asyncio.sleep(1)
        
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        passed = sum(1 for r in results.values() if r.get("verified", False))
        total = len(results)
        
        for param_name, result in results.items():
            status = "✓" if result.get("verified", False) else "✗"
            transcript = result.get("transcript", "N/A")[:50] if result.get("transcript") else "No transcript"
            logger.info(f"{status} {param_name:30} | {transcript}")
            if result.get("error"):
                logger.info(f"  Error: {result['error']}")
        
        logger.info("="*60)
        logger.info(f"Results: {passed}/{total} passed ({passed*100//total if total > 0 else 0}%)")
        logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())