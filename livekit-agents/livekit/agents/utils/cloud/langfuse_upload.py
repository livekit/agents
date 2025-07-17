import os
import requests
import base64
import hashlib
import uuid
from typing import Optional
from datetime import datetime, timezone

from ...log import logger


def upload_wav_to_langfuse(wav_bytes: bytes, trace_id: int) -> str:
    """
    Upload WAV file bytes to Langfuse cloud storage and return a formatted media string.
    
    Args:
        wav_bytes: WAV file content as bytes
        
    Returns:
        Formatted string: "@@@langfuseMedia:type={MIME_TYPE}|id={LANGFUSE_MEDIA_ID}|source={SOURCE_TYPE}@@@"
        
    Raises:
        ValueError: If required environment variables are not set
        requests.RequestException: If upload fails
    """
    # Get required environment variables
    base_url = os.getenv("LANGFUSE_HOST")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    
    if not all([base_url, public_key, secret_key]):
        raise ValueError(
            "Missing required environment variables: LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY"
        )
    
    # Prepare upload parameters
    content_type = "audio/wav"
    content_sha256 = base64.b64encode(hashlib.sha256(wav_bytes).digest()).decode()
    content_length = len(wav_bytes)
    field = "input"  # Can be "input", "output", or "metadata"
    
    # Create upload URL request body
    create_upload_url_body = {
        "traceId": str(trace_id),
        "contentType": content_type,
        "contentLength": content_length,
        "sha256Hash": content_sha256,
        "field": field,
    }
    
    # Request upload URL from Langfuse
    try:
        upload_url_request = requests.post(
            f"{base_url}/api/public/media",
            auth=(public_key, secret_key),
            headers={"Content-Type": "application/json"},
            json=create_upload_url_body,
        )
        upload_url_request.raise_for_status()
        upload_url_response = upload_url_request.json()
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to get upload URL from Langfuse: {e}")
    
    # Get media ID
    media_id = upload_url_response.get("mediaId")
    if not media_id:
        raise ValueError("No media ID returned from Langfuse")
    
    # Upload file if upload URL is provided (file not already uploaded)
    upload_url = upload_url_response.get("uploadUrl")
    logger.info(f"upload_url: {upload_url}")
    
    if upload_url:
        try:
            upload_response = None
            upload_response = requests.put(
                upload_url,
                headers={
                    "Content-Type": content_type,
                    "x-amz-checksum-sha256": content_sha256,
                },
                data=wav_bytes,
            )
            upload_response.raise_for_status()
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to upload file to Langfuse: {e}")
        
        if upload_response is not None:
          requests.patch(
              f"{base_url}/api/public/media/{upload_url_response['mediaId']}",
              auth=(public_key or "", secret_key or ""),
              headers={"Content-Type": "application/json"},
              json={
                  "uploadedAt": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), # ISO 8601
                  "uploadHttpStatus": upload_response.status_code,
                  "uploadHttpError": upload_response.text if upload_response.status_code != 200 else None,
              },
          )
    
    
    # Return formatted media string
    return f"@@@langfuseMedia:type={content_type}|id={media_id}|source=bytes@@@"


def upload_audio_to_langfuse(
    audio_bytes: bytes,
    content_type: str = "audio/wav",
    field: str = "input"
) -> str:
    """
    Upload audio file bytes to Langfuse cloud storage and return a formatted media string.
    More flexible version that supports different audio formats.
    
    Args:
        audio_bytes: Audio file content as bytes
        content_type: MIME type of the audio file (default: "audio/wav")
        field: Field type for the upload ("input", "output", or "metadata")
        
    Returns:
        Formatted string: "@@@langfuseMedia:type={MIME_TYPE}|id={LANGFUSE_MEDIA_ID}|source={SOURCE_TYPE}@@@"
    """
    # Get required environment variables
    base_url = os.getenv("LANGFUSE_HOST")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    
    if not all([base_url, public_key, secret_key]):
        raise ValueError(
            "Missing required environment variables: LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY"
        )
    
    # Prepare upload parameters
    content_sha256 = base64.b64encode(hashlib.sha256(audio_bytes).digest()).decode()
    trace_id = str(uuid.uuid4())
    content_length = len(audio_bytes)
    
    # Create upload URL request body
    create_upload_url_body = {
        "traceId": trace_id,
        "contentType": content_type,
        "contentLength": content_length,
        "sha256Hash": content_sha256,
        "field": field,
    }
    
    # Request upload URL from Langfuse
    try:
        upload_url_request = requests.post(
            f"{base_url}/api/public/media",
            auth=(public_key, secret_key),
            headers={"Content-Type": "application/json"},
            json=create_upload_url_body,
        )
        upload_url_request.raise_for_status()
        upload_url_response = upload_url_request.json()
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to get upload URL from Langfuse: {e}")
    
    # Get media ID
    media_id = upload_url_response.get("mediaId")
    if not media_id:
        raise ValueError("No media ID returned from Langfuse")
    
    # Upload file if upload URL is provided (file not already uploaded)
    upload_url = upload_url_response.get("uploadUrl")
    if upload_url:
        try:
            upload_response = requests.put(
                upload_url,
                headers={
                    "Content-Type": content_type,
                    "x-amz-checksum-sha256": content_sha256,
                },
                data=audio_bytes,
            )
            upload_response.raise_for_status()
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to upload file to Langfuse: {e}")
    
    # Return formatted media string
    return f"@@@langfuseMedia:type={content_type}|id={media_id}|source=bytes@@@"
