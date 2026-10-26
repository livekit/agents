# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalized BCP-47 language identifier."""

from __future__ import annotations

from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from . import _core


class LanguageCode(str):
    """Normalized BCP-47 language identifier. Accepts any common format.

    Examples::

        LanguageCode("english")  # → "en"
        LanguageCode("eng")      # → "en"
        LanguageCode("en")       # → "en"
        LanguageCode("en-US")    # → "en-US"
        LanguageCode("en_us")    # → "en-US"
        LanguageCode("multi")    # → "multi"
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
        return _core.base_language(self)  # type: ignore[no-any-return]

    @property
    def iso(self) -> str:
        """ISO 639-1 tag with region, e.g., ``'zh-CN'`` from ``'cmn-Hans-CN'``."""
        return _core.language_iso(self)  # type: ignore[no-any-return]

    @property
    def region(self) -> str | None:
        """Region code, e.g., ``'US'`` from ``'en-US'``, or ``None``."""
        return _core.language_region(self)  # type: ignore[no-any-return]

    def to_language_name(self) -> str | None:
        """Return the English language name (e.g. ``'english'``), or ``None`` if unknown."""
        return _core.language_name(self)  # type: ignore[no-any-return]
