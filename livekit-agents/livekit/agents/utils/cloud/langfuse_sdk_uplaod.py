import base64

def upload_wav_to_langfuse_media(wav_data: bytes, langfuse_client):
    try:
      from langfuse.media import LangfuseMedia

      base_64_string = base64.b64encode(wav_data).decode('utf-8')
      base_64_data_uri = f"data:audio/wav;base64,{base_64_string}"
      wrapped_obj = LangfuseMedia(
          base64_data_uri=base_64_data_uri
      )

                      
      langfuse_client.update_current_trace(
          output={
              "context": wrapped_obj
          }
      )
    except ImportError:
        raise Exception("Langfuse SDK is not installed")
    except Exception as e:
       raise Exception(f"Failed to upload wav to langfuse media: {e}")

