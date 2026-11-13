from __future__ import annotations

import os

from livekit.agents import http, tokenize, tts
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .models import FluxTTSModels
from .tts import _DeepgramTTS, _TTSOptions

# Flux TTS is served on the /v2/speak endpoint over both WebSocket (streaming) and
# REST (batch). A single https base URL works for both: `to_deepgram_url` flips the
# scheme to wss when a WebSocket connection is requested.
# https://developers.deepgram.com/docs/flux-tts/overview
BASE_URL_V2 = "https://api.deepgram.com/v2/speak"


class TTSv2(_DeepgramTTS):
    def __init__(
        self,
        *,
        model: FluxTTSModels | str = "flux-alexis-en",
        encoding: str = "linear16",
        sample_rate: int = 24000,
        bit_rate: int | None = None,
        api_key: str | None = None,
        base_url: str = BASE_URL_V2,
        http_session: http.HttpSession | None = None,
        mip_opt_out: bool = False,
    ) -> None:
        """
        Create a new instance of Deepgram Flux TTS (the /v2/speak endpoint).

        Args:
            model (FluxTTSModels | str): Flux TTS model to use. Defaults to "flux-alexis-en".
                Model names follow the `flux-{voice}-{language}` format.
                See https://developers.deepgram.com/docs/flux-tts/overview for available voices.
            encoding (str): Audio encoding to use. Defaults to "linear16". stream() supports
                "linear16", "mulaw" and "alaw"; compressed encodings (mp3/opus/flac/aac) are
                available on the batch synthesize() path only.
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            bit_rate (int | None): Bit rate for compressed encodings (e.g. mp3). Defaults to None.
                Applies to the batch synthesize() path only.
            api_key (str): Deepgram API key. If not provided, will look for DEEPGRAM_API_KEY in environment.
            base_url (str): Base URL for Deepgram Flux TTS API. Defaults to "https://api.deepgram.com/v2/speak"
            http_session (http.HttpSession): Optional HTTP session to use for requests.
            mip_opt_out (bool): Opt out of the Deepgram Model Improvement Program. Defaults to
                False (requests may be used to improve models). See https://dpgr.am/deepgram-mip

        """  # noqa: E501
        api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("Deepgram API key required. Set DEEPGRAM_API_KEY or provide api_key.")

        super().__init__(
            _TTSOptions(
                model=model,
                encoding=encoding,
                sample_rate=sample_rate,
                bit_rate=bit_rate,
                base_url=base_url,
                api_key=api_key,
                mip_opt_out=mip_opt_out,
                flux=True,
            ),
            http_session,
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[FluxTTSModels | str] = NOT_GIVEN,
        encoding: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        bit_rate: NotGivenOr[int | None] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            model (FluxTTSModels | str): Flux TTS model to use.
            encoding (str): Audio encoding to use.
            sample_rate (int): Sample rate of audio in Hz.
            bit_rate (int | None): Bit rate for compressed encodings (e.g. mp3).
            sentence_tokenizer (tokenize.SentenceTokenizer): What splits the text into the chunks
                the TTS is sent.
            pacing (tts.SentenceStreamPacer | None): A ``SentenceStreamPacer``, or ``None`` to send
                each chunk as soon as it is split.
        """
        super().update_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing)
        self._set_options(
            model=model, encoding=encoding, sample_rate=sample_rate, bit_rate=bit_rate
        )
