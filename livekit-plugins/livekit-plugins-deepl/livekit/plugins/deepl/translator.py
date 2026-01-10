from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable
from typing import Literal

import deepl
from deepl import Formality, Language, SplitSentences, Translator as DeepLTranslator
from deepl.api_data import ModelType
from livekit.agents.plugin import Plugin

from .log import logger
from .version import __version__

# DeepL API supports 'free' and 'pro' tiers with different endpoints
DeepLTier = Literal["free", "pro"]


class DeepLTranslationPlugin(Plugin):
    """
    A LiveKit Agent plugin for real-time text translation using DeepL.
    """

    def __init__(self, debug: bool = False) -> None:
        super().__init__(__name__, __version__, __package__, logger)

        auth_key = os.environ.get("DEEPL_AUTH_KEY")
        if not auth_key:
            raise ValueError(
                "DEEPL_AUTH_KEY environment variable not set. Please provide your DeepL API key."
            )

        deepl_logger = logging.getLogger("deepl")
        if debug:
            deepl_logger.setLevel(logging.DEBUG)
        else:
            deepl_logger.setLevel(logging.WARNING)

        server_url = os.environ.get("DEEPL_SERVER_URL")
        self._tier = "free" if server_url == DeepLTranslator._DEEPL_SERVER_URL_FREE else "pro"

        self._translator = deepl.Translator(auth_key, server_url=server_url)
        logger.info(f"DeepLTranslationPlugin initialized for {self._tier} tier.")

    async def translate_text(
        self,
        text: str | Iterable[str],
        *,
        source_lang: str | Language | None = None,
        target_lang: str | Language,
        context: str | None = None,
        split_sentences: str | SplitSentences | None = None,
        preserve_formatting: bool | None = None,
        formality: str | Formality | None = None,
        tag_handling: str | None = None,
        tag_handling_version: str | None = None,
        outline_detection: bool | None = None,
        non_splitting_tags: str | list[str] | None = None,
        splitting_tags: str | list[str] | None = None,
        ignore_tags: str | list[str] | None = None,
        model_type: str | ModelType | None = None,
        custom_instructions: list[str] | None = None,
        extra_body_parameters: dict | None = None,
    ) -> list[str]:
        """
        Translate text(s) into the target language using DeepL API.

        Args:
            text: The text to translate. Can be a string or iterable of strings.
            source_lang: The source language code (e.g., "en", "fr", "de").
                        Use None for automatic source language detection.
            target_lang: The target language code (e.g., "en-US", "fr", "de").
            context: Additional context that influences translation but isn't translated.
                    Characters in context are not counted toward billing.
            split_sentences: Controls sentence splitting behavior.
                            "0" = no splitting, "1" = split on punctuation and newlines,
                            "nonewlines" = split on punctuation only.
            preserve_formatting: When True, preserves original formatting instead of correcting it.
            formality: Desired formality level ("default", "more", "less",
                    "prefer_more", "prefer_less"). Only available for certain languages.
            tag_handling: Type of tags to handle ("xml" or "html").
            tag_handling_version: Tag handling algorithm version ("v1" or "v2").
            outline_detection: When False, disables automatic XML structure detection.
            non_splitting_tags: XML tags that never split sentences (comma-separated or list).
            splitting_tags: XML tags that always split sentences (comma-separated or list).
            ignore_tags: XML tags containing text not to translate (comma-separated or list).
            model_type: Translation model type ("quality_optimized", "latency_optimized").
            custom_instructions: List of custom instructions (max 10, each max 300 chars).
                            Only available for certain languages.
            extra_body_parameters: Additional key/value pairs for the API request body.

        Returns:
            The translated text as a string or list of strings.

        Raises:
            ValueError: If translation fails, invalid language is provided, or
                    glossaries are used with auto-detection.
            deepl.exceptions.DeepLException: For DeepL-specific API errors.

        Note:
            - Beta languages (enabled via extra_body_parameters) don't support
            formality or glossaries and require quality_optimized models.
            - Custom instructions require quality_optimized models and are only
            available for specific languages.
        """
        if not text or (isinstance(text, str) and not text.strip()):
            return []

        if source_lang == target_lang:
            return [text] if isinstance(text, str) else list(text)

        # The deepl library's translate_text is synchronous.
        # We use asyncio.to_thread to run it in a separate thread pool
        # to avoid blocking the event loop.
        try:
            result = await asyncio.to_thread(
                self._translator.translate_text,
                text,
                source_lang=source_lang,
                target_lang=target_lang,
                context=context,
                split_sentences=split_sentences,
                preserve_formatting=preserve_formatting,
                formality=formality,
                tag_handling=tag_handling,
                tag_handling_version=tag_handling_version,
                outline_detection=outline_detection,
                non_splitting_tags=non_splitting_tags,
                splitting_tags=splitting_tags,
                ignore_tags=ignore_tags,
                model_type=model_type,
                custom_instructions=custom_instructions,
                extra_body_parameters=extra_body_parameters,
            )

            if isinstance(result, list):
                return [r.text for r in result]
            else:
                return [result.text]

        except deepl.exceptions.DeepLException as e:
            raise ValueError(f"DeepL API error: {e}") from e
        except Exception as e:
            raise ValueError(f"Translation failed: {e}") from e

    @property
    def provider(self) -> str:
        return "DeepL"

    @property
    def model(self) -> str:
        return f"DeepL-{self._tier}"
