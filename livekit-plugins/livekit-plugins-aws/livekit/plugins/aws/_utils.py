import os
from typing import Literal

import boto3


def _get_aws_credentials(
    api_key: str | None, api_secret: str | None, region: str | None
):
    region = region or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError(
            "AWS_DEFAULT_REGION must be set using the argument or by setting the AWS_DEFAULT_REGION environment variable."
        )

    # If API key and secret are provided, create a session with them
    if api_key and api_secret:
        session = boto3.Session(
            aws_access_key_id=api_key,
            aws_secret_access_key=api_secret,
            region_name=region,
        )
    else:
        # Use default credentials from environment or AWS config
        session = boto3.Session(region_name=region)

    # Validate if session credentials are available
    credentials = session.get_credentials()
    if not credentials or not credentials.access_key or not credentials.secret_key:
        raise ValueError("No valid AWS credentials found.")
    return credentials.access_key, credentials.secret_key


TTS_SPEECH_ENGINE = Literal["standard", "neural", "long-form", "generative"]
TTS_LANGUAGE = Literal[
    "arb",
    "cmn-CN",
    "cy-GB",
    "da-DK",
    "de-DE",
    "en-AU",
    "en-GB",
    "en-GB-WLS",
    "en-IN",
    "en-US",
    "es-ES",
    "es-MX",
    "es-US",
    "fr-CA",
    "fr-FR",
    "is-IS",
    "it-IT",
    "ja-JP",
    "hi-IN",
    "ko-KR",
    "nb-NO",
    "nl-NL",
    "pl-PL",
    "pt-BR",
    "pt-PT",
    "ro-RO",
    "ru-RU",
    "sv-SE",
    "tr-TR",
    "en-NZ",
    "en-ZA",
    "ca-ES",
    "de-AT",
    "yue-CN",
    "ar-AE",
    "fi-FI",
    "en-IE",
    "nl-BE",
    "fr-BE",
    "cs-CZ",
    "de-CH",
]

TTS_OUTPUT_FORMAT = Literal["mp3", "pcm"]
