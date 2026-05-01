"""
Audio processor for Trillet Voiceguard plugin
Processes audio to 16-bit PCM, Mono, 16kHz format as required by Voiceguard API
"""
import numpy as np
import asyncio
from typing import Optional, Callable
from livekit import rtc
from .log import logger


class VoiceguardAudioProcessor:
    """
    Processes audio data from LiveKit room for Voiceguard voice detection
    Converts to required format: 16-bit PCM, Mono, 16kHz sample rate
    """
    
    def __init__(self, room: rtc.Room):
        self.room = room
        self.audio_callbacks = []
        self.is_processing = False
        
        # Voiceguard audio format requirements
        self.target_sample_rate = 16000  # 16kHz
        self.target_channels = 1  # Mono
        self.target_dtype = np.int16  # 16-bit PCM
        
        # LiveKit audio format (will be determined from frames)
        self.livekit_sample_rate = None  # Will be set from first frame
        self.livekit_channels = None     # Will be set from first frame
        
    def add_audio_callback(self, callback: Callable[[bytes, dict], None]):
        """Add a callback function to receive processed audio data"""
        self.audio_callbacks.append(callback)
        
    def remove_audio_callback(self, callback: Callable[[bytes, dict], None]):
        """Remove a callback function"""
        if callback in self.audio_callbacks:
            self.audio_callbacks.remove(callback)
            
    async def start_processing(self):
        """Start processing audio from room participants"""
        if self.is_processing:
            logger.warning("Voiceguard audio processing already started")
            return
            
        logger.info("Starting Voiceguard audio processing (16-bit PCM, Mono, 16kHz)")
        self.is_processing = True
        
        # Set up audio track handlers for remote participants (incoming audio)
        self.room.on("track_subscribed", self._on_track_subscribed)
        
        # Process existing participants
        logger.info(f"Processing {len(self.room.remote_participants)} existing participants")
        for participant in self.room.remote_participants.values():
            logger.info(f"Setting up Voiceguard audio for existing participant: {participant.identity}")
            await self._setup_participant_audio(participant)
            
    async def stop_processing(self):
        """Stop processing audio"""
        logger.info("Stopping Voiceguard audio processing")
        self.is_processing = False
        
        # Clear callbacks immediately to prevent further processing
        self.audio_callbacks.clear()
        
        # Give a small delay to allow running streams to see the is_processing flag
        await asyncio.sleep(0.1)
        
    async def _setup_participant_audio(self, participant: rtc.RemoteParticipant):
        """Set up audio processing for a specific participant"""
        logger.debug(f"Setting up Voiceguard audio for participant: {participant.identity}")
        
        for publication in participant.track_publications.values():
            if publication.kind == rtc.TrackKind.KIND_AUDIO and publication.track:
                await self._process_audio_track(publication.track, participant)
                
    def _on_track_subscribed(self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Handle new audio tracks"""
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.debug(f"New audio track from participant for Voiceguard: {participant.identity}")
            asyncio.create_task(self._process_audio_track(track, participant))
            
    async def _process_audio_track(self, track: rtc.AudioTrack, participant: rtc.RemoteParticipant):
        """Process audio data from a track and convert to Voiceguard format"""
        logger.debug(f"Processing audio track for Voiceguard from {participant.identity}")
        
        try:
            # Create audio frame reader
            audio_stream = rtc.AudioStream(track)
            logger.info(f"Created audio stream for Voiceguard from {participant.identity}")
            
            frame_count = 0
            async for event in audio_stream:
                frame_count += 1
                if not self.is_processing:
                    logger.info(f"Voiceguard processing stopped after {frame_count} events")
                    break
                
                # Extract actual frame from AudioFrameEvent
                if hasattr(event, 'frame'):
                    frame = event.frame
                else:
                    logger.warning(f"Event has no frame attribute: {type(event)}")
                    continue
                    
                # Set source audio properties from frame if not already set
                if self.livekit_sample_rate is None:
                    self.livekit_sample_rate = frame.sample_rate
                    logger.info(f"Detected LiveKit audio format: {frame.sample_rate}Hz, {frame.num_channels} channels")
                if self.livekit_channels is None:
                    self.livekit_channels = frame.num_channels
                
                # Process raw audio data directly 
                raw_audio_bytes = self._extract_and_convert_raw_audio(frame)
                
                if raw_audio_bytes:
                    # Notify callbacks with raw bytes
                    metadata = {
                        "participant_identity": participant.identity,
                        "sample_rate": self.target_sample_rate,
                        "channels": self.target_channels,
                        "format": "16-bit PCM"
                    }
                    
                    self._notify_audio_callbacks(raw_audio_bytes, metadata)
                else:
                    logger.warning("Failed to extract raw audio from frame")
                        
        except Exception as e:
            logger.error(f"Error processing audio track for Voiceguard: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    def _extract_and_convert_raw_audio(self, frame: rtc.AudioFrame) -> Optional[bytes]:
        """Extract raw audio from LiveKit AudioFrame and convert to Voiceguard format"""
        try:
            # Get raw audio data from frame
            audio_data = np.frombuffer(frame.data, dtype=np.int16)
            
            if len(audio_data) == 0:
                logger.warning("Empty audio frame data")
                return None
            
            # Handle multi-channel to mono conversion (optimized)
            if frame.num_channels == 2:
                # Stereo to mono: use in-place operations for better performance
                audio_data = audio_data.reshape(-1, 2)
                audio_data = ((audio_data[:, 0].astype(np.int32) + audio_data[:, 1].astype(np.int32)) // 2).astype(np.int16)
            elif frame.num_channels > 2:
                # Multi-channel to mono: take every Nth sample (first channel)
                audio_data = audio_data[::frame.num_channels]
            
            # Downsample if needed (e.g., 48kHz -> 16kHz) - optimized
            if frame.sample_rate != self.target_sample_rate:
                downsample_ratio = frame.sample_rate // self.target_sample_rate
                if downsample_ratio > 1:
                    audio_data = audio_data[::downsample_ratio]
            
            # Convert back to raw bytes (16-bit PCM as expected by Voiceguard)
            return audio_data.tobytes()
            
        except Exception as e:
            logger.error(f"Error extracting raw audio from frame: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def _notify_audio_callbacks(self, audio_data: bytes, metadata: dict):
        """Notify all registered callbacks with new audio data"""
        # Don't notify callbacks if processing has stopped
        if not self.is_processing or not self.audio_callbacks:
            return
            
        for callback in self.audio_callbacks:
            try:
                callback(audio_data, metadata)
            except Exception as e:
                logger.error(f"Error in Voiceguard audio callback: {e}") 