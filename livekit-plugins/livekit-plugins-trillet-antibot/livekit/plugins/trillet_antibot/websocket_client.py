"""
WebSocket client for Trillet Voiceguard plugin
Sends raw binary audio data and receives JSON detection results
"""
import asyncio
import websockets
import json
import logging
from typing import Optional, Any
from .log import logger

# Suppress verbose websocket binary data logging
logging.getLogger('websockets.client').setLevel(logging.WARNING)
logging.getLogger('websockets.server').setLevel(logging.WARNING)
logging.getLogger('websockets').setLevel(logging.WARNING)


class VoiceguardWebsocketClient:
    """
    WebSocket client for Trillet Voiceguard voice detection service
    """
    
    def __init__(self, websocket_url: str):
        self.websocket_url = websocket_url
        self.websocket = None
        self.is_connected = False
        self.receive_task = None
        self.voiceguard_streamer = None  # Will be set by parent
        
    def set_streamer(self, streamer):
        """Set reference to parent streamer for result callbacks"""
        self.voiceguard_streamer = streamer
        
    async def connect(self):
        """Connect to the Voiceguard websocket server"""
        try:
            logger.info(f"Connecting to Voiceguard websocket: {self.websocket_url}")
            self.websocket = await websockets.connect(self.websocket_url)
            self.is_connected = True
            
            # Start receiving responses
            self.receive_task = asyncio.create_task(self._receive_responses())
            
            logger.info("Successfully connected to Voiceguard websocket")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Voiceguard websocket: {e}")
            self.is_connected = False
            return False
            
    async def disconnect(self):
        """Disconnect from the Voiceguard websocket server"""
        logger.info("Disconnecting from Voiceguard websocket")
        
        if self.receive_task and not self.receive_task.done():
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                logger.debug("Voiceguard websocket receive task was cancelled during shutdown - this is expected")
        
        if self.websocket and self.is_connected:
            try:
                await self.websocket.close()
                logger.info("Disconnected from Voiceguard websocket")
            except Exception as e:
                logger.error(f"Error disconnecting from Voiceguard websocket: {e}")
        
        self.is_connected = False
        self.websocket = None
        
    async def send_audio_bytes(self, audio_data: bytes):
        """Send raw binary audio data to Voiceguard websocket"""
        if not self.is_connected or not self.websocket:
            return False
            
        try:
            # Send raw binary bytes (no JSON wrapping as per API spec)
            await self.websocket.send(audio_data)
            return True
            
        except websockets.exceptions.ConnectionClosed as e:
            # Only log once when connection closes to prevent spam
            if self.is_connected:  # Only log if we weren't already aware it was closed
                if "Maximum duration" in str(e.reason):
                    logger.debug(f"Voiceguard connection closed as expected: {e.reason}")
                else:
                    logger.warning(f"Voiceguard connection closed unexpectedly: {e.code} {e.reason}")
                self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to send audio data to Voiceguard: {e}")
            return False
            
    async def _receive_responses(self):
        """Continuously receive and process detection responses from Voiceguard"""
        logger.info("Starting Voiceguard response listener")
        try:
            while self.is_connected and self.websocket:
                try:
                    # Receive response message
                    message = await self.websocket.recv()
                    
                    # Parse JSON response
                    try:
                        result = json.loads(message)
                        
                        # Check if this is a final summary or individual segment result
                        if result.get("type") == "final_summary":
                            # Handle final summary
                            if self.voiceguard_streamer and hasattr(self.voiceguard_streamer, 'on_final_summary'):
                                self.voiceguard_streamer.on_final_summary(result)
                            logger.info(f"Received Voiceguard final summary: {result.get('overall_prediction', 'unknown')} prediction")
                        else:
                            # Handle individual segment result
                            if self.voiceguard_streamer and hasattr(self.voiceguard_streamer, 'on_detection_result'):
                                self.voiceguard_streamer.on_detection_result(result)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Voiceguard response as JSON: {e}")
                        logger.debug(f"Raw message: {message}")
                        
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(f"Voiceguard websocket connection closed: {e.code} {e.reason}")
                    # Immediately stop connection and notify streamer
                    self.is_connected = False
                    if hasattr(self, 'voiceguard_streamer') and self.voiceguard_streamer:
                        asyncio.create_task(self.voiceguard_streamer._on_websocket_closed())
                    break
                except Exception as e:
                    logger.error(f"Error receiving Voiceguard response: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in Voiceguard response listener: {e}")
        finally:
            logger.info("Voiceguard response listener stopped")
            self.is_connected = False
        
    def _validate_response(self, result: dict) -> bool:
        """Validate the Voiceguard individual segment response format"""
        required_fields = ["prediction", "confidence_fake", "confidence_real"]
        
        # Check required fields exist
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing required field '{field}' in Voiceguard response")
                return False
                
        # Validate prediction value
        prediction = result.get("prediction")
        if prediction not in ["FAKE", "REAL"]:
            logger.warning(f"Invalid prediction value: {prediction}")
            return False
            
        # Validate confidence scores are numeric
        try:
            float(result["confidence_fake"])
            float(result["confidence_real"])
        except (ValueError, TypeError):
            logger.warning("Confidence scores are not numeric")
            return False
            
        return True 
        
    def _validate_final_summary(self, result: dict) -> bool:
        """Validate the Voiceguard final summary response format"""
        required_fields = [
            "type", "total_segments", "fake_predictions", "real_predictions", 
            "overall_prediction", "average_fake_confidence", "average_real_confidence", "fake_percentage"
        ]
        
        # Check required fields exist
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing required field '{field}' in Voiceguard final summary")
                return False
                
        # Validate type is final_summary
        if result.get("type") != "final_summary":
            logger.warning(f"Invalid type in final summary: {result.get('type')}")
            return False
            
        # Validate overall prediction value
        overall_prediction = result.get("overall_prediction")
        if overall_prediction not in ["FAKE", "REAL"]:
            logger.warning(f"Invalid overall prediction value: {overall_prediction}")
            return False
            
        # Validate numeric fields
        numeric_fields = ["total_segments", "fake_predictions", "real_predictions", 
                         "average_fake_confidence", "average_real_confidence", "fake_percentage"]
        for field in numeric_fields:
            try:
                float(result[field])
            except (ValueError, TypeError):
                logger.warning(f"Field '{field}' is not numeric in final summary")
                return False
                
        return True 