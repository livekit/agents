# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the declarations in:
#     rust/crates/lk-agents-abi/src/tokenize/language/code.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from .. import _core


class LanguageCode(str):
    """Normalized BCP-47 language identifier. Accepts any common format.

    Examples:

    ```python
    LanguageCode("english")  # -> "en"
    LanguageCode("eng")      # -> "en"
    LanguageCode("en")       # -> "en"
    LanguageCode("en-US")    # -> "en-US"
    LanguageCode("en_us")    # -> "en-US"
    LanguageCode("multi")    # -> "multi"
    ```
    """

    def __new__(cls, code: str) -> LanguageCode:
        return super().__new__(cls, _core.normalize_language(code))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls,
            serialization=core_schema.to_string_ser_schema(),
        )

    @property
    def language(self) -> str:
        """Base language code (ISO 639-1 when possible).

        E.g., ``'en'`` from ``'en-US'``, ``'zh'`` from ``'cmn-Hans-CN'``.
        """
        return _core.base_language(self)

    @property
    def iso(self) -> str:
        """ISO 639-1 tag with region, e.g., ``'zh-CN'`` from ``'cmn-Hans-CN'``."""
        return _core.language_iso(self)

    @property
    def region(self) -> str | None:
        """Region code, e.g., ``'US'`` from ``'en-US'``, or ``None``."""
        return _core.language_region(self)

    def to_language_name(self) -> str | None:
        """Return the English language name (e.g. ``'english'``), or ``None`` if unknown."""
        return _core.language_name(self)


LanguageCode.__module__ = "livekit.agents"
