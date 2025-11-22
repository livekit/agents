from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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


class BargeinRequest(BaseModel):
    """Request model for bargein detection."""

    jobId: str = Field(..., description="Job ID from LiveKit")
    workerId: str = Field(..., description="Worker ID from LiveKit")
    waveform: str = Field(..., description="Base64-encoded audio waveform (float32)")
    agentId: str | None = Field(None, description="Optional agent ID")
    threshold: float = Field(0.95, description="Threshold for bargein detection")
    min_frames: int = Field(2, description="Minimum number of frames for bargein detection")
    created_at: float = Field(..., description="Timestamp of the audio waveform")


class BargeinResponse(BaseModel):
    """Response model for bargein detection."""

    is_bargein: bool = Field(..., description="Whether bargein is detected")
    confidence: float | None = Field(None, description="Confidence score (optional)")
    created_at: float = Field(..., description="Timestamp of the data creation")


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


def decode_waveform(encoded_waveform: str) -> np.ndarray:
    """
    Decode base64-encoded waveform to numpy array.

    Args:
        encoded_waveform: Base64-encoded audio waveform

    Returns:
        Numpy array of float32 audio samples
    """
    waveform_bytes = base64.b64decode(encoded_waveform)
    waveform = np.frombuffer(waveform_bytes, dtype=np.float32)
    return waveform.reshape(1, -1)


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
async def detect_bargein(request: BargeinRequest) -> BargeinResponse:
    """
    Detect bargein in audio waveform.

    Args:
        request: BargeinRequest containing job info and audio waveform

    Returns:
        BargeinResponse with detection result
    """
    logger.info(
        f"Received bargein detection request for job={request.jobId}, worker={request.workerId}"
    )

    if onnx_session is None:
        raise HTTPException(
            status_code=503,
            detail="ONNX model not loaded. Check server logs for details.",
        )

    try:
        # Decode the waveform
        waveform = decode_waveform(request.waveform)

        # Run inference
        is_bargein = run_inference(waveform, request.threshold, request.min_frames)
        if is_bargein:
            import soundfile as sf

            sf.write("bargein.wav", waveform.T, 16000)

        logger.info(f"Bargein detection result: is_bargein={is_bargein}")

        return BargeinResponse(
            is_bargein=is_bargein, confidence=None, created_at=request.created_at
        )

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
    - Client sends: {"type": "session.create", "settings": {"sample_rate": "16000"}}
    - Server sends: {"type": "session.created"}
    - Client sends: {"type": "input_audio", "audio": "<base64>", "sample_rate": 16000, "num_channels": 1, "threshold": 0.95, "min_frames": 2}
    - Server sends: {"type": "bargein_detected"} (when detected)
    - Client sends: {"type": "session.finalize"}
    - Server sends: {"type": "session.finalized"}
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    if onnx_session is None:
        await websocket.send_json({"type": "error", "message": "ONNX model not loaded"})
        await websocket.close()
        return

    try:
        # Wait for session.create message
        while True:
            try:
                data = await websocket.receive_text()
                msg = json.loads(data)
                msg_type = msg.get("type")

                if msg_type == "session.create":
                    logger.info("Session created")
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

        # Process audio frames
        while True:
            try:
                data = await websocket.receive_text()
                msg = json.loads(data)
                msg_type = msg.get("type")

                if msg_type == "input_audio":
                    # Decode and process audio

                    audio_b64 = msg.get("audio")
                    threshold = msg.get("threshold", 0.95)
                    min_frames = msg.get("min_frames", 2)
                    created_at = msg.get("created_at", time.time())

                    if not audio_b64:
                        await websocket.send_json(
                            {"type": "error", "message": "Missing audio data"}
                        )
                        continue

                    try:
                        waveform = decode_waveform(audio_b64)
                        is_bargein = run_inference(waveform, threshold, min_frames)
                        delta = time.time() - created_at

                        await websocket.send_json(
                            {
                                "type": "inference_done",
                                "delta": delta,
                                "is_bargein": is_bargein,
                                "created_at": created_at,
                            }
                        )

                        if is_bargein:
                            logger.info("Bargein detected via WebSocket")
                            await websocket.send_json({"type": "bargein_detected"})

                    except Exception as e:
                        logger.error(f"Error processing audio: {e}", exc_info=True)
                        await websocket.send_json(
                            {"type": "error", "message": f"Error processing audio: {str(e)}"}
                        )

                elif msg_type == "session.finalize":
                    logger.info("Session finalized")
                    await websocket.send_json({"type": "session.finalized"})
                    break

                else:
                    logger.warning(f"Unknown message type: {msg_type}")
                    await websocket.send_json(
                        {"type": "error", "message": f"Unknown message type: {msg_type}"}
                    )

            except json.JSONDecodeError as e:
                await websocket.send_json({"type": "error", "message": f"Invalid JSON: {str(e)}"})
                continue

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
