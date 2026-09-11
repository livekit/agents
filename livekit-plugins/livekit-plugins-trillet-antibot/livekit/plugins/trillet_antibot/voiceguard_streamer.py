"""
Trillet Antibot Plugin - AI voice detection for calls
"""
import asyncio
import time
import json
import io
import wave
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
import aiohttp
from .log import logger
from .websocket_client import VoiceguardWebsocketClient
from .audio_processor import VoiceguardAudioProcessor
from livekit import api


class TrilletAntiBot:
    """
    Trillet AntiBot plugin for AI voice detection during calls

    Features:
    - Real-time voice analysis during call
    - Optional call termination on fake voice detection
    - Audio recording and S3 upload for archival
    - Configurable confidence thresholds
    
    Usage:
        # Basic usage
        antibot = TrilletAntiBot(room, call_context, ctx, ai_key="your-key", workspace_id="your-workspace-id")

        # With S3 audio recording
        antibot = TrilletAntiBot(
            room, call_context, ctx, 
            ai_key="your-key",
            workspace_id="your-workspace-id",
            save_audio_to_s3=True,
            terminate_on_fake=True
        )

        await antibot.start_streaming()
    """
    
    def __init__(self, room, ctx, ai_key: str, workspace_id: str, duration_seconds: int = 30, segment_length: float = 3.0, confidence_threshold: float = 0.70, terminate_on_fake: bool = False, save_audio_to_s3: bool = False):
        self.room = room
        self.ctx = ctx
        self.ai_key = ai_key
        self.workspace_id = workspace_id
        self.duration_seconds = duration_seconds
        self.segment_length = segment_length
        self.confidence_threshold = confidence_threshold
        self.terminate_on_fake = terminate_on_fake
        self.save_audio_to_s3 = save_audio_to_s3
        self.is_streaming = False
        self.stream_task = None
        self.start_time = None
        self.detection_results: List[Dict[str, Any]] = []
        self.final_summary: Optional[Dict[str, Any]] = None
        self.call_termination_reason: Optional[str] = None
        self.final_summary_received = False

        # Audio recording for S3 upload
        self.recorded_audio_chunks = []  # Store audio chunks for S3 upload
        self.audio_sample_rate = 16000  # 16kHz
        self.audio_channels = 1  # Mono
        self.audio_sample_width = 2  # 16-bit = 2 bytes
        
        # WebSocket URL with segment length, max duration, confidence threshold, API key, and workspace ID parameters
        base_url = "wss://p01--trillet-voice-guard-dev--j629vb9mq7pk.ccvhxjx8pb.code.run/trillet-voiceguard/ws/detect"
        self.websocket_url = f"{base_url}?segment_length={segment_length}&max_duration={duration_seconds}&confidence_threshold={confidence_threshold}&api_key={ai_key}&workspace_id={workspace_id}"
        
        # Initialize components
        self.websocket_client = VoiceguardWebsocketClient(self.websocket_url)
        self.websocket_client.set_streamer(self)  # Set callback reference
        
        self.audio_processor = VoiceguardAudioProcessor(room)
        
    async def start_streaming(self):
        """Start the voiceguard streaming for 30 seconds"""
        if self.is_streaming:
            logger.warning("Voiceguard streaming already in progress")
            return
            
        logger.info("Starting Trillet Voiceguard streaming")
        self.is_streaming = True
        self.start_time = time.time()\
        
        
        # Start the main detection task
        self.stream_task = asyncio.create_task(self._run_voiceguard_detection())
        
    async def stop_streaming(self):
        """Stop the voiceguard streaming"""
        if not self.is_streaming:
            return
            
        logger.info("Stopping Trillet Voiceguard streaming")
        self.is_streaming = False
        
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                logger.debug("Voiceguard stream task was cancelled during shutdown - this is expected")
                
        # Clean up connections - but let server close WebSocket first to receive final summary
        await self.audio_processor.stop_processing()
        # Ensure callback is removed to prevent further audio processing
        self.audio_processor.remove_audio_callback(self._on_audio_data)
        
        # Only disconnect if we already received the final summary, or after waiting
        if not self.final_summary_received:
            logger.info("Waiting for final summary before disconnecting...")
            # Wait up to 5 seconds for final summary
            for i in range(50):  # 50 * 0.1s = 5 seconds
                if self.final_summary_received:
                    logger.info("Final summary received, proceeding with disconnect")
                    break
                await asyncio.sleep(0.1)
            
            if not self.final_summary_received:
                logger.warning("Final summary not received after 5s wait, forcing disconnect")
        
        # Now disconnect WebSocket
        await self.websocket_client.disconnect()
        
        # Upload audio to S3 after WebSocket closes

        
        # Log final results
        await self._log_final_results()
        
    def _on_audio_data(self, audio_data: bytes, metadata: dict):
        """Callback for processed audio data - send to voiceguard websocket and store for S3"""
        if not self.is_streaming or not self.websocket_client.is_connected:
            return
        
        # Log audio data being sent for debugging
        logger.debug(f"Sending audio chunk: {len(audio_data)} bytes, format: {metadata.get('format', 'unknown')}")
        
        # Store audio chunk for S3 upload if enabled
        if self.save_audio_to_s3:
            self.recorded_audio_chunks.append(audio_data)
        
        # Send raw audio bytes to websocket (no JSON wrapping)
        asyncio.create_task(self.websocket_client.send_audio_bytes(audio_data))
        
    async def _on_websocket_closed(self):
        """Handle websocket closure - immediately stop audio processing"""
        logger.info("Websocket closed - stopping audio processing immediately")
        
        # Immediately stop streaming and audio processing
        self.is_streaming = False

        if self.save_audio_to_s3:
            # Upload recorded audio to S3 via server endpoint
            await self._upload_audio_to_s3()
        
        # Remove audio callback to prevent further processing
        try:
            self.audio_processor.remove_audio_callback(self._on_audio_data)
            await self.audio_processor.stop_processing()
        except Exception as e:
            logger.debug(f"Error stopping audio processor: {e}")
            
        # Cancel the main stream task if running
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()
        
    async def _run_voiceguard_detection(self):
        """Main voiceguard detection loop"""
        try:
            # Connect to websocket
            logger.info("Starting Voiceguard detection process")
            if not await self.websocket_client.connect():
                logger.error("Failed to connect to Voiceguard websocket - aborting detection")
                return
                
            # Set up audio processing callback
            self.audio_processor.add_audio_callback(self._on_audio_data)
            
            # Start audio processing
            await self.audio_processor.start_processing()
            logger.info(f"Voiceguard detection running for {self.duration_seconds} seconds (server will auto-close)")
            
            # Let the server handle the timeout and auto-close
            # We'll run until the websocket disconnects or our task is cancelled
            start_time = time.time()
            last_log_time = start_time
            
            while self.is_streaming:
                await asyncio.sleep(1.0)  # Check every second
                
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Log progress every 10 seconds
                if current_time - last_log_time >= 10:
                    logger.info(f"Voiceguard detection progress: {elapsed:.0f}s elapsed, {len(self.detection_results)} results received")
                    last_log_time = current_time
                
                # Safety timeout (a bit longer than server timeout)
                if elapsed > self.duration_seconds + 10:
                    logger.warning("Voiceguard detection exceeded expected duration - forcing stop")
                    break
                
                # Check if websocket is still connected
                if not self.websocket_client.is_connected:
                    logger.info("Voiceguard websocket disconnected - stopping detection")
                    break
                
            logger.info(f"Voiceguard detection completed. Total duration: {time.time() - start_time:.1f}s, Results: {len(self.detection_results)}")
            await self.stop_streaming()
            
        except asyncio.CancelledError:
            logger.info("Trillet Voiceguard streaming cancelled")
        except Exception as e:
            logger.error(f"Error in Trillet Voiceguard streaming: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.is_streaming = False
        
    def on_detection_result(self, result: dict):
        """Handle detection result from websocket"""
        self.detection_results.append({
            "timestamp": time.time(),
            "elapsed_time": time.time() - self.start_time if self.start_time else 0,
            **result
        })
        
        prediction = result.get("prediction", "UNKNOWN")
        confidence_fake = result.get("confidence_fake", 0)
        confidence_real = result.get("confidence_real", 0)

        # # Check for FAKE prediction and conditionally terminate call
        # if prediction == "FAKE":
        #     logger.info(f"FAKE voice detected with confidence {confidence_fake}.")
        #     if self.terminate_on_fake:
        #         logger.info("Terminating call due to terminate_on_fake=True.")
        #         self.call_termination_reason = "fake_voice_detected"
        #         asyncio.create_task(self._terminate_call())
        #     else:
        #         logger.info("Call termination disabled (terminate_on_fake=False). Continuing detection.")
        
        # Log detection result for analysis
        logger.info(f"Voice detection result: {prediction} (fake: {confidence_fake:.3f}, real: {confidence_real:.3f})")
    async def _terminate_call(self):
        """Terminate the call by deleting the room."""
        logger.info(f"Terminating call in room: {self.room.name} due to fake voice detection")
        try:
            api_client = self.ctx.api
            await api_client.room.delete_room(api.DeleteRoomRequest(room=self.room.name))
            logger.info(f"Call terminated due to fake voice detection in room: {self.room.name}")
        except Exception as e:
            logger.error(f"Error terminating call: {e}")
        
    def on_final_summary(self, summary: dict):
        """Handle final summary from Voiceguard service"""
        logger.info("=== VOICEGUARD FINAL SUMMARY ===")
        logger.info(f"Overall Prediction: {summary.get('overall_prediction', 'UNKNOWN')}")
        logger.info(f"Total Segments: {summary.get('total_segments', 0)}")
        logger.info(f"FAKE Predictions: {summary.get('fake_predictions', 0)}")
        logger.info(f"REAL Predictions: {summary.get('real_predictions', 0)}")
        logger.info(f"Fake Percentage: {summary.get('fake_percentage', 0)}%")
        logger.info(f"Average FAKE Confidence: {summary.get('average_fake_confidence', 0):.3f}")
        logger.info(f"Average REAL Confidence: {summary.get('average_real_confidence', 0):.3f}")
        logger.info("==============================")

        # Conditionally terminate call if final summary indicates FAKE
        if summary.get("overall_prediction") == "FAKE":
            logger.info(f"Final summary indicates FAKE voice with confidence {summary.get('average_fake_confidence')}.")
            if self.terminate_on_fake:
                logger.info("Terminating call due to terminate_on_fake=True.")
                self.call_termination_reason = "fake_voice_detected"
                asyncio.create_task(self._terminate_call())
            else:
                logger.info("Call termination disabled (terminate_on_fake=False). Detection complete.")
        
        # Store final summary for retrieval
        self.final_summary = {
            "timestamp": time.time(),
            "elapsed_time": time.time() - self.start_time if self.start_time else 0,
            **summary
        }
        
        # Mark that we received the final summary
        self.final_summary_received = True

    async def _log_final_results(self):
        """Log final voiceguard analysis results - now relies on official summary from service"""
        if not self.detection_results:
            logger.warning("No voiceguard detection results received")
            return
            
        # Check if we received an official final summary
        if self.final_summary:
            logger.info("Official Voiceguard final summary received and logged above")
        else:
            logger.warning("No official final summary received from Voiceguard service")
            # Fallback to our own calculation if no official summary received
            fake_predictions = [r for r in self.detection_results if r.get("prediction") == "FAKE"]
            real_predictions = [r for r in self.detection_results if r.get("prediction") == "REAL"]
            
            avg_fake_confidence = sum(r.get("confidence_fake", 0) for r in self.detection_results) / len(self.detection_results)
            avg_real_confidence = sum(r.get("confidence_real", 0) for r in self.detection_results) / len(self.detection_results)
            
            logger.info(f"Fallback Summary - Total segments: {len(self.detection_results)}, FAKE: {len(fake_predictions)}, REAL: {len(real_predictions)}")
            
            # Store fallback summary
            self.final_summary = {
                "total_segments": len(self.detection_results),
                "fake_predictions": len(fake_predictions),
                "real_predictions": len(real_predictions),
                "avg_fake_confidence": avg_fake_confidence,
                "avg_real_confidence": avg_real_confidence,
                "duration_seconds": self.duration_seconds,
                "segment_length": self.segment_length,
                "is_fallback": True
            }
        
    async def _upload_audio_to_s3(self):
        """Upload recorded audio chunks to S3 via server endpoint"""
        if not self.recorded_audio_chunks:
            logger.warning("No audio chunks recorded, skipping S3 upload")
            return
        
        try:
            # Create WAV file from audio chunks
            wav_buffer = self._create_wav_file()
            if not wav_buffer:
                logger.error("Failed to create WAV file from audio chunks")
                return
            
            # Generate filename with timestamp and room info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            room_name = getattr(self.room, 'name', 'unknown_room')
            filename = f"voiceguard_audio_{room_name}_{timestamp}.wav"
            
            # Prepare upload URL (assuming same base as websocket but HTTP)
            upload_url = "https://p01--trillet-voice-guard-dev--j629vb9mq7pk.ccvhxjx8pb.code.run/upload-audio-simple"
            
            # Upload to server
            async with aiohttp.ClientSession() as session:
                # Prepare form data
                data = aiohttp.FormData()
                data.add_field('audio_file', 
                             wav_buffer.getvalue(), 
                             filename=filename,
                             content_type='audio/wav')
                data.add_field('api_key', self.ai_key)
                data.add_field('workspace_id', self.workspace_id)
                
                logger.info(f"Uploading audio to S3: {filename} ({len(wav_buffer.getvalue())} bytes)")
                
                async with session.post(upload_url, data=data) as response:
                    response_data = await response.json()
                    
                    if response.status == 200 and response_data.get("status") == 1:
                        upload_details = response_data.get("upload_details", {})
                        s3_url = upload_details.get("s3_url")
                        logger.info(f"Audio successfully uploaded to S3: {s3_url}")
                        
                        # Store S3 info in final summary for retrieval
                        if self.final_summary:
                            self.final_summary["audio_upload"] = {
                                "success": True,
                                "s3_url": s3_url,
                                "s3_key": upload_details.get("s3_key"),
                                "filename": filename,
                                "uploaded_at": upload_details.get("uploaded_at"),
                                "file_size": len(wav_buffer.getvalue())
                            }
                    else:
                        error_msg = response_data.get("error", "Unknown upload error")
                        logger.error(f"Failed to upload audio to S3: {error_msg}")
                        
                        # Store error info in final summary
                        if self.final_summary:
                            self.final_summary["audio_upload"] = {
                                "success": False,
                                "error": error_msg,
                                "filename": filename
                            }
                            
        except Exception as e:
            logger.error(f"Error uploading audio to S3: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Store error info in final summary
            if self.final_summary:
                self.final_summary["audio_upload"] = {
                    "success": False,
                    "error": str(e),
                    "filename": f"voiceguard_audio_{getattr(self.room, 'name', 'unknown_room')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                }
    
    def _create_wav_file(self) -> Optional[io.BytesIO]:
        """Create a WAV file from recorded audio chunks"""
        if not self.recorded_audio_chunks:
            return None
            
        try:
            # Create BytesIO buffer for WAV file
            wav_buffer = io.BytesIO()
            
            # Create WAV file
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.audio_channels)  # Mono
                wav_file.setsampwidth(self.audio_sample_width)  # 16-bit = 2 bytes
                wav_file.setframerate(self.audio_sample_rate)  # 16kHz
                
                # Write all audio chunks
                for chunk in self.recorded_audio_chunks:
                    wav_file.writeframes(chunk)
            
            # Reset buffer position to beginning
            wav_buffer.seek(0)
            
            logger.info(f"Created WAV file: {len(self.recorded_audio_chunks)} chunks, {wav_buffer.getbuffer().nbytes} bytes total")
            return wav_buffer
            
        except Exception as e:
            logger.error(f"Error creating WAV file: {e}")
            return None
        
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all detection results"""
        return self.detection_results.copy()
    
    def get_final_summary(self) -> Optional[Dict[str, Any]]:
        """Get the final summary from Voiceguard service"""
        return self.final_summary.copy() if self.final_summary else None
    
    def get_call_termination_reason(self) -> Optional[str]:
        """Get the reason for call termination, if any"""
        return self.call_termination_reason
    
    def get_audio_upload_status(self) -> Optional[Dict[str, Any]]:
        """Get the audio upload status and details"""
        if self.final_summary and "audio_upload" in self.final_summary:
            return self.final_summary["audio_upload"].copy()
        return None
        
    def is_voice_likely_fake(self, threshold: float = 0.7) -> bool:
        """Determine if voice is likely fake based on detection results"""
        if not self.detection_results:
            return False
            
        # Simple majority vote with confidence threshold
        high_confidence_fake = sum(1 for r in self.detection_results 
                                  if r.get("prediction") == "FAKE" and r.get("confidence_fake", 0) > threshold)
        
        return high_confidence_fake > len(self.detection_results) / 2 