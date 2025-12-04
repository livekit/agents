from __future__ import annotations

import json
import logging
import os
import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

onnx_session: ort.InferenceSession | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global onnx_session
    model_path = os.getenv("BARGEIN_MODEL_PATH", "bargein_model.onnx")

    try:
        onnx_session = load_onnx_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        logger.warning("Server will start but inference will fail until model is loaded")

    yield

    logger.info("Shutting down bargein detection server")


app = FastAPI(
    title="Bargein Detection API",
    description="API for detecting bargein in audio waveforms using ONNX model",
    version="1.0.0",
    lifespan=lifespan,
)


class BargeinResponse(BaseModel):
    """Response model for bargein detection."""

    is_bargein: bool = Field(..., description="Whether bargein is detected")
    confidence: float | None = Field(None, description="Confidence score (optional)")
    created_at: int = Field(..., description="Timestamp of the data creation (nanoseconds)")


def load_onnx_model(model_path: str) -> ort.InferenceSession:
    """
    Load the ONNX model for bargein detection.

    Args:
        model_path: Path to the ONNX model file

    Returns:
        ONNX inference session
    """
    logger.info(f"Loading ONNX model from {model_path}")
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],  # Use CPU by default
    )
    logger.info("ONNX model loaded successfully")
    logger.info(f"Model inputs: {[input.name for input in session.get_inputs()]}")
    logger.info(f"Model outputs: {[output.name for output in session.get_outputs()]}")
    return session


def decode_waveform_s16le(raw_bytes: bytes) -> np.ndarray:
    """
    Decode raw s16le bytes to numpy array (float32 normalized).

    Args:
        raw_bytes: Raw audio bytes in s16le format

    Returns:
        Numpy array of float32 audio samples, shape (1, num_samples)
    """
    waveform_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
    # Normalize to float32 in range [-1, 1]
    waveform_f32 = waveform_i16.astype(np.float32) / np.iinfo(np.int16).max
    return waveform_f32.reshape(1, -1)


def run_inference(waveform: np.ndarray, threshold: float = 0.95, min_frames: int = 2) -> bool:
    """
    Run inference on the waveform using the ONNX model.

    Args:
        waveform: Preprocessed audio waveform

    Returns:
        Tuple of (is_bargein, confidence_score)
    """
    if onnx_session is None:
        raise RuntimeError("ONNX model not loaded")

    # Get input name from the model
    input_name = onnx_session.get_inputs()[0].name

    # Run inference
    scores = onnx_session.run(None, {input_name: waveform})[0]
    probs = scores > threshold
    running_true_counts = np.convolve(probs.astype(int), np.ones(min_frames), mode="valid")
    return bool(np.any(running_true_counts >= min_frames))


@app.get("/")
async def root() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Bargein Detection API is running",
        "model_loaded": onnx_session is not None,
    }


@app.post("/bargein", response_model=BargeinResponse)
async def detect_bargein(
    request: Request,
    threshold: float = Query(0.65, description="Threshold for bargein detection"),
    min_frames: int = Query(1, description="Minimum number of frames for bargein detection"),
    created_at: int = Query(..., description="Timestamp in nanoseconds"),
) -> BargeinResponse:
    """
    Detect bargein in audio waveform.

    Expects raw s16le audio bytes in the request body with Content-Type: application/octet-stream.
    Metadata is passed via URL query parameters.

    Args:
        request: FastAPI request with raw audio bytes in body
        threshold: Detection threshold (0.0-1.0)
        min_frames: Minimum consecutive frames above threshold
        created_at: Timestamp in nanoseconds for latency tracking

    Returns:
        BargeinResponse with detection result
    """
    if onnx_session is None:
        raise HTTPException(
            status_code=503,
            detail="ONNX model not loaded. Check server logs for details.",
        )

    try:
        # Read raw bytes from body
        raw_bytes = await request.body()

        # Decode s16le waveform
        waveform = decode_waveform_s16le(raw_bytes)

        # Run inference
        is_bargein = run_inference(waveform, threshold, min_frames)
        if is_bargein:
            import soundfile as sf

            sf.write("bargein.wav", waveform.T, 16000)

        logger.info(f"Bargein detection result: is_bargein={is_bargein}")

        return BargeinResponse(is_bargein=is_bargein, confidence=None, created_at=created_at)

    except Exception as e:
        logger.error(f"Error during bargein detection: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error during bargein detection: {str(e)}",
        ) from None


@app.websocket("/bargein")
async def websocket_bargein(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for bargein detection.

    Protocol:
    - Client sends (text): {"type": "session.create", "settings": {"sample_rate": 16000, "encoding": "s16le", "threshold": 0.65, "min_frames": 1}}
    - Server sends (text): {"type": "session.created"}
    - Client sends (binary): 8-byte header (uint64 LE timestamp in ns) + s16le audio bytes
    - Server sends (text): {"type": "bargein_detected", "created_at": <timestamp>} (when detected)
    - Client sends (text): {"type": "session.close"}
    - Server sends (text): {"type": "session.closed"}
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    if onnx_session is None:
        await websocket.send_json({"type": "error", "message": "ONNX model not loaded"})
        await websocket.close()
        return

    # Session settings (defaults)
    threshold = 0.65
    min_frames = 1

    try:
        # Wait for session.create message
        while True:
            try:
                data = await websocket.receive_text()
                msg = json.loads(data)
                msg_type = msg.get("type")

                if msg_type == "session.create":
                    settings = msg.get("settings", {})
                    threshold = settings.get("threshold", 0.65)
                    min_frames = settings.get("min_frames", 1)
                    encoding = settings.get("encoding", "s16le")
                    logger.info(
                        f"Session created: threshold={threshold}, min_frames={min_frames}, encoding={encoding}"
                    )
                    await websocket.send_json({"type": "session.created"})
                    break
                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Expected session.create, got {msg_type}"}
                    )
                    await websocket.close()
                    return
            except json.JSONDecodeError as e:
                await websocket.send_json({"type": "error", "message": f"Invalid JSON: {str(e)}"})
                await websocket.close()
                return

        # Process audio frames (binary) and control messages (text)
        while True:
            msg = await websocket.receive()

            # Handle binary audio frames
            if "bytes" in msg and msg["bytes"] is not None:
                try:
                    raw_data = msg["bytes"]

                    # Parse 8-byte header (uint64 LE timestamp in nanoseconds)
                    if len(raw_data) < 8:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Binary message too short (missing header)",
                            }
                        )
                        continue

                    (created_at,) = struct.unpack("<Q", raw_data[:8])
                    audio_bytes = raw_data[8:]

                    if len(audio_bytes) == 0:
                        await websocket.send_json(
                            {"type": "error", "message": "No audio data after header"}
                        )
                        continue

                    # Decode s16le waveform
                    waveform = decode_waveform_s16le(audio_bytes)
                    is_bargein = run_inference(waveform, threshold, min_frames)

                    if is_bargein:
                        logger.info("Bargein detected via WebSocket")
                        await websocket.send_json(
                            {"type": "bargein_detected", "created_at": created_at}
                        )

                except Exception as e:
                    logger.error(f"Error processing audio: {e}", exc_info=True)
                    await websocket.send_json(
                        {"type": "error", "message": f"Error processing audio: {str(e)}"}
                    )

            # Handle text control messages
            elif "text" in msg and msg["text"] is not None:
                try:
                    data = json.loads(msg["text"])
                    msg_type = data.get("type")

                    if msg_type == "session.close":
                        logger.info("Session closed")
                        await websocket.send_json({"type": "session.closed"})
                        break
                    else:
                        logger.warning(f"Unknown text message type: {msg_type}")
                        await websocket.send_json(
                            {"type": "error", "message": f"Unknown message type: {msg_type}"}
                        )

                except json.JSONDecodeError as e:
                    await websocket.send_json(
                        {"type": "error", "message": f"Invalid JSON: {str(e)}"}
                    )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": f"Internal error: {str(e)}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/health")
async def health() -> dict[str, Any]:
    """Detailed health check endpoint."""
    return {
        "status": "healthy" if onnx_session is not None else "degraded",
        "model_loaded": onnx_session is not None,
        "model_info": {
            "inputs": [input.name for input in onnx_session.get_inputs()] if onnx_session else [],
            "outputs": [output.name for output in onnx_session.get_outputs()]
            if onnx_session
            else [],
        }
        if onnx_session
        else None,
    }


if __name__ == "__main__":
    import uvicorn  # type: ignore

    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
