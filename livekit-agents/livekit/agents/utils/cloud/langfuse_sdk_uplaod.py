import base64

from langfuse._client.client import Langfuse
from opentelemetry import trace


def upload_wav_to_langfuse_media(wav_data: bytes, langfuse_client: Langfuse, current_span):
    try:
        from langfuse.media import LangfuseMedia

        base_64_string = base64.b64encode(wav_data).decode("utf-8")
        base_64_data_uri = f"data:audio/wav;base64,{base_64_string}"

        wrapped_obj = LangfuseMedia(base64_data_uri=base_64_data_uri)

        with trace.use_span(current_span, end_on_exit=False):
            langfuse_client.update_current_span(output={"context": wrapped_obj})
    except ImportError as import_err:
        raise Exception("Langfuse SDK is not installed") from import_err
    except Exception as e:
        raise Exception(f"Failed to upload wav to langfuse media: {e}") from e
