import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Union, BinaryIO
from io import BytesIO

from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import ContentSettings
from azure.core.exceptions import AzureError

from livekit import rtc
from ...log import logger


class AzureBlobUploader:
    """Azure Blob Storage uploader for audio files from TTS streams"""
    
    def __init__(
        self,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        connection_string: Optional[str] = None,
        sas_token: Optional[str] = None,
        container_name: Optional[str] = None
    ):
        """
        Initialize Azure Blob Storage uploader
        
        Args:
            account_name: Azure storage account name (or set AZURE_STORAGE_ACCOUNT_NAME env var)
            account_key: Azure storage account key (or set AZURE_STORAGE_ACCOUNT_KEY env var)
            connection_string: Full connection string (or set AZURE_STORAGE_CONNECTION_STRING env var)
            sas_token: SAS token for authentication (or set AZURE_STORAGE_SAS_TOKEN env var)
            container_name: Container name to upload files to (or set AZURE_STORAGE_CONTAINER_NAME env var)
        """
        
        # Get credentials from environment if not provided
        self.account_name = account_name or os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = account_key or os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.sas_token = sas_token or os.getenv("AZURE_STORAGE_SAS_TOKEN")
        self.container_name = container_name or os.getenv("AZURE_STORAGE_CONTAINER_NAME")
        
        # Validate authentication options
        if not any([self.connection_string, (self.account_name and self.account_key), 
                   (self.account_name and self.sas_token)]):
            raise ValueError(
                "Azure authentication requires one of:\n"
                "1. connection_string (or AZURE_STORAGE_CONNECTION_STRING env var)\n"
                "2. account_name + account_key (or AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY env vars)\n"
                "3. account_name + sas_token (or AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_SAS_TOKEN env vars)"
            )
        
        self._client: Optional[BlobServiceClient] = None
    
    async def _get_client(self) -> BlobServiceClient:
        """Get or create Azure Blob Service client"""
        if self._client is None:
            if self.connection_string:
                self._client = BlobServiceClient.from_connection_string(self.connection_string)
            elif self.account_key:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(account_url=account_url, credential=self.account_key)
            elif self.sas_token:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(account_url=account_url, credential=self.sas_token)
        
        return self._client
    
    async def upload_audio_bytes(
        self,
        audio_data: Union[bytes, BinaryIO],
        blob_name: str,
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = True,
        content_type: str = "audio/wav"
    ) -> str:
        """
        Upload raw audio bytes to Azure Blob Storage
        
        Args:
            audio_data: Raw audio bytes or file-like object
            blob_name: Name for the blob
            metadata: Optional metadata to attach to the blob
            overwrite: Whether to overwrite existing blob
            content_type: Content type for the blob
        
        Returns:
            The blob URL
        """
        try:
            client = await self._get_client()
            
            # Ensure container exists
            try:
                container_client = client.get_container_client(self.container_name)
                await container_client.create_container()
                logger.info(f"Created container: {self.container_name}")
            except Exception:
                # Container already exists or we don't have permission to create it
                pass
            
            # Prepare content settings
            content_settings = ContentSettings(content_type=content_type)
            
            # Upload the blob
            blob_client = client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            logger.info(f"Uploading audio to blob: {blob_name}")
            
            await blob_client.upload_blob(
                data=audio_data,
                content_settings=content_settings,
                metadata=metadata,
                overwrite=overwrite
            )
            
            # Get the blob URL
            blob_url = blob_client.url
            logger.info(f"Successfully uploaded audio to: {blob_url}")
            
            return blob_url
            
        except AzureError as e:
            logger.error(f"Azure error uploading blob {blob_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading blob {blob_name}: {e}")
            raise
    
    async def upload_audio_frame(
        self,
        audio_frame: rtc.AudioFrame,
        blob_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = True,
        content_type: str = "audio/wav"
    ) -> str:
        """
        Upload a single AudioFrame to Azure Blob Storage
        
        Args:
            audio_frame: The AudioFrame to upload
            blob_name: Name for the blob (auto-generated if None)
            metadata: Optional metadata to attach to the blob
            overwrite: Whether to overwrite existing blob
            content_type: Content type for the blob
        
        Returns:
            The blob URL
        """
        try:
            # Convert to WAV bytes
            wav_data = audio_frame.to_wav_bytes()
            
            # Generate blob name if not provided
            if blob_name is None:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                blob_name = f"audio_frame_{timestamp}_{unique_id}.wav"
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "source": "livekit-audio-frame",
                "sample_rate": str(audio_frame.sample_rate),
                "num_channels": str(audio_frame.num_channels),
                "duration_seconds": str(audio_frame.duration),
                "samples_per_channel": str(audio_frame.samples_per_channel),
                "upload_timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return await self.upload_audio_bytes(
                audio_data=wav_data,
                blob_name=blob_name,
                metadata=metadata,
                overwrite=overwrite,
                content_type=content_type
            )
            
        except Exception as e:
            logger.error(f"Failed to upload AudioFrame to Azure Blob Storage: {e}")
            raise
    
    async def close(self):
        """Close the Azure Blob Service client"""
        if self._client:
            await self._client.close()
            self._client = None


# Convenience function for simple uploads
# async def upload_tts_to_azure(
#     chunked_stream: ChunkedStream,
#     container_name: str = "audio-files",
#     blob_name: Optional[str] = None,
#     metadata: Optional[Dict[str, str]] = None,
#     account_name: Optional[str] = None,
#     account_key: Optional[str] = None,
#     connection_string: Optional[str] = None,
#     sas_token: Optional[str] = None,
#     overwrite: bool = True
# ) -> str:
#     """
#     Convenience function to upload TTS ChunkedStream to Azure Blob Storage
    
#     Args:
#         chunked_stream: The TTS ChunkedStream containing audio data
#         container_name: Container name to upload to
#         blob_name: Name for the blob (auto-generated if None)
#         metadata: Optional metadata to attach to the blob
#         account_name: Azure storage account name (or set AZURE_STORAGE_ACCOUNT_NAME env var)
#         account_key: Azure storage account key (or set AZURE_STORAGE_ACCOUNT_KEY env var)
#         connection_string: Full connection string (or set AZURE_STORAGE_CONNECTION_STRING env var)
#         sas_token: SAS token for authentication (or set AZURE_STORAGE_SAS_TOKEN env var)
#         overwrite: Whether to overwrite existing blob
    
#     Returns:
#         The blob URL
    
#     Example:
#         ```python
#         # Using environment variables for auth
#         blob_url = await upload_tts_to_azure(
#             chunked_stream=tts_stream,
#             container_name="my-audio-files",
#             blob_name="speech_output.wav"
#         )
        
#         # Using explicit credentials
#         blob_url = await upload_tts_to_azure(
#             chunked_stream=tts_stream,
#             account_name="mystorageaccount",
#             account_key="your_account_key",
#             container_name="audio-files"
#         )
#         ```
#     """
#     uploader = AzureBlobUploader(
#         account_name=account_name,
#         account_key=account_key,
#         connection_string=connection_string,
#         sas_token=sas_token,
#         container_name=container_name
#     )
    
#     try:
#         return await uploader.upload_chunked_stream(
#             chunked_stream=chunked_stream,
#             blob_name=blob_name,
#             metadata=metadata,
#             overwrite=overwrite
#         )
#     finally:
#         await uploader.close()
