from __future__ import annotations

import re
import typing
from collections.abc import Callable

from ..utils import aio, shortuuid
from .tokenizer import SentenceStream, TokenData, WordStream

# Tokenizers can either provide us with a list of tokens or a list of tokens along with their start and end indices.  # noqa: E501
# If the start and end indices are not available, we attempt to locate the token within the text using str.find.  # noqa: E501
TokenizeCallable = Callable[[str], list[str] | list[tuple[str, int, int]]]

# the tag name must start with a letter so "<5>" / "<3 wins>" are not counted as
# tags — this keeps the depth counter consistent with the letter-start tail check
# in _has_unclosed_xml_tags (all TTS markup tags are letter-named)
_XML_TAG_RE = re.compile(r"<(/?)([A-Za-z]\w*)[^>]*?(/?)\s*>")


def _has_unclosed_xml_tags(text: str) -> bool:
    """Return True if *text* contains an incomplete or unclosed XML tag."""
    if "<" not in text:
        return False

    # incomplete tag at end: a tag-shaped "<" without a matching ">". Only "<"
    # followed by a name start ("/" or a letter) is tag-shaped — a bare "<" as in
    # "3 < 5" or "<3" is plain text and must not hold up streaming. Text ending
    # exactly at "<" is treated as tag-shaped: the next chunk resolves it.
    last_open = text.rfind("<")
    last_close = text.rfind(">")
    if last_open > last_close:
        nxt = text[last_open + 1 : last_open + 2]
        if not nxt or nxt == "/" or nxt.isalpha():
            return True

    # unbalanced open/close pairs
    depth = 0
    for m in _XML_TAG_RE.finditer(text):
        is_closing = m.group(1) == "/"
        is_self_closing = m.group(3) == "/"
        if is_self_closing:
            continue
        elif is_closing:
            depth -= 1
        else:
            depth += 1

    return depth > 0


def _is_xml_only(text: str) -> bool:
    """Return True if *text* contains XML tags but no substantive text content."""
    if "<" not in text:
        return False

    stripped = _XML_TAG_RE.sub("", text).strip()
    return len(stripped) == 0


def _clean_to_orig(clean_pos: int, tag_spans: list[tuple[int, int]]) -> int:
    """Map a position in tag-stripped text to the corresponding original position.

    Tags that sit right at the boundary are left for the next sentence.
    """
    orig = clean_pos
    for tag_start, tag_end in tag_spans:
        if tag_start < orig:
            orig += tag_end - tag_start
        else:
            break
    return orig


def _xml_wrap_tokenizer(
    tokenize_fnc: TokenizeCallable,
) -> TokenizeCallable:
    """Wrap a tokenizer so XML tags don't interfere with sentence splitting.

    Strips tag markers before tokenization (content inside wrapping tags is
    kept so the tokenizer can account for its length), remaps offsets back to
    the original text, and merges sentences with unclosed or tag-only content.
    """

    def _wrapped(text: str) -> list[str] | list[tuple[str, int, int]]:
        try:
            return _wrapped_impl(text)
        except Exception:
            return [(text, 0, len(text))] if text.strip() else []

    def _wrapped_impl(text: str) -> list[str] | list[tuple[str, int, int]]:
        tag_spans = [(m.start(), m.end()) for m in _XML_TAG_RE.finditer(text)]
        if not tag_spans:
            return tokenize_fnc(text)

        clean_text = _XML_TAG_RE.sub("", text)
        if not clean_text.strip():
            return [(text, 0, len(text))] if text.strip() else []

        raw_tokens = tokenize_fnc(clean_text)
        if not raw_tokens:
            return []

        # extract clean-text end offsets
        clean_ends: list[int] = []
        for tok in raw_tokens:
            clean_ends.append(tok[2] if isinstance(tok, tuple) else -1)

        # if tokenizer didn't provide offsets, approximate from token lengths
        if clean_ends[0] == -1:
            pos = 0
            clean_ends = []
            for tok in raw_tokens:
                tok_text = tok if isinstance(tok, str) else tok[0]
                idx = clean_text.find(tok_text, pos)
                pos = (idx if idx >= 0 else pos) + len(tok_text)
                clean_ends.append(pos)

        # remap to original positions and rebuild sentences
        result: list[tuple[str, int, int]] = []
        start = 0
        for clean_end in clean_ends:
            orig_end = _clean_to_orig(clean_end, tag_spans)
            sentence = text[start:orig_end].strip()
            if sentence:
                result.append((sentence, start, orig_end))
            start = orig_end

        if start < len(text):
            sentence = text[start:].strip()
            if sentence:
                result.append((sentence, start, len(text)))

        # merge sentences with unclosed tags or tag-only content
        if result:
            merged: list[tuple[str, int, int]] = [result[0]]
            for sent_text, s_start, s_end in result[1:]:
                prev_text, prev_start, _ = merged[-1]
                if _has_unclosed_xml_tags(prev_text) or _is_xml_only(prev_text):
                    merged[-1] = (text[prev_start:s_end].strip(), prev_start, s_end)
                else:
                    merged.append((sent_text, s_start, s_end))
            result = merged

        return result

    return _wrapped


class BufferedTokenStream:
    def __init__(
        self,
        *,
        tokenize_fnc: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
        max_token_len: int | None = None,
        retain_format: bool = False,
        xml_aware: bool = False,
    ) -> None:
        """
        Args:
            xml_aware: treat XML markup as atomic — never split a tag across tokens
                and merge tag-only/unclosed spans forward. Only enable when the input
                actually carries markup (e.g. expressive TTS): a stray "<" in plain
                text can otherwise hold back streaming until flush.
        """
        self._event_ch = aio.Chan[TokenData]()
        self._tokenize_fnc = _xml_wrap_tokenizer(tokenize_fnc) if xml_aware else tokenize_fnc
        self._min_ctx_len = min_ctx_len
        self._min_token_len = min_token_len
        self._max_token_len = max_token_len
        self._retain_format = retain_format
        self._xml_aware = xml_aware
        self._current_segment_id = shortuuid()

        self._buf_tokens: list[str] = []  # <= min_token_len
        self._in_buf = ""
        self._out_buf = ""

    @typing.no_type_check
    def push_text(self, text: str) -> None:
        self._check_not_closed()
        if not text:
            return
        self._in_buf += text

        if len(self._in_buf) < self._min_ctx_len:
            return

        while True:
            tokens = self._tokenize_fnc(self._in_buf)
            if len(tokens) <= 1:
                break

            tok = tokens[0]
            tok_text = tok[0] if isinstance(tok, tuple) else tok

            # don't emit a token that would split an XML tag
            if self._xml_aware and _has_unclosed_xml_tags(tok_text):
                break

            tokens.pop(0)

            # if adding this sentence would exceed max, emit what we have first
            if (
                self._max_token_len
                and self._out_buf
                and len(self._out_buf) + 1 + len(tok_text) > self._max_token_len
            ):
                self._event_ch.send_nowait(
                    TokenData(token=self._out_buf, segment_id=self._current_segment_id)
                )
                self._out_buf = ""

            if self._out_buf:
                self._out_buf += " "

            self._out_buf += tok_text
            if len(self._out_buf) >= self._min_token_len:
                self._event_ch.send_nowait(
                    TokenData(token=self._out_buf, segment_id=self._current_segment_id)
                )

                self._out_buf = ""

            if isinstance(tok, tuple):
                self._in_buf = self._in_buf[tok[2] :]
            else:
                tok_i = max(self._in_buf.find(tok), 0)
                self._in_buf = self._in_buf[tok_i + len(tok) :].lstrip()

    @typing.no_type_check
    def flush(self) -> None:
        self._check_not_closed()

        if self._in_buf or self._out_buf:
            tokens = self._tokenize_fnc(self._in_buf)
            for tok in tokens:
                tok_text = tok[0] if isinstance(tok, tuple) else tok

                # honor the cap here too: appending everything into one chunk could
                # exceed max_token_len and trip a provider's send limit. Emit the
                # buffer before it would overflow, then keep batching the rest.
                if (
                    self._max_token_len
                    and self._out_buf
                    and len(self._out_buf) + 1 + len(tok_text) > self._max_token_len
                ):
                    self._event_ch.send_nowait(
                        TokenData(token=self._out_buf, segment_id=self._current_segment_id)
                    )
                    self._out_buf = ""

                if self._out_buf:
                    self._out_buf += " "
                self._out_buf += tok_text

            if self._out_buf:
                self._event_ch.send_nowait(
                    TokenData(token=self._out_buf, segment_id=self._current_segment_id)
                )

        self._current_segment_id = shortuuid()
        self._in_buf = ""
        self._out_buf = ""

    def end_input(self) -> None:
        self.flush()
        self._event_ch.close()

    async def aclose(self) -> None:
        self._event_ch.close()

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def __aiter__(self) -> BufferedTokenStream:
        return self

    async def __anext__(self) -> TokenData:
        return await self._event_ch.__anext__()


class BufferedSentenceStream(BufferedTokenStream, SentenceStream):
    def __init__(
        self,
        *,
        tokenizer: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
        max_token_len: int | None = None,
        xml_aware: bool = False,
    ) -> None:
        super().__init__(
            tokenize_fnc=tokenizer,
            min_token_len=min_token_len,
            min_ctx_len=min_ctx_len,
            max_token_len=max_token_len,
            xml_aware=xml_aware,
        )


class BufferedWordStream(BufferedTokenStream, WordStream):
    def __init__(
        self,
        *,
        tokenizer: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
    ) -> None:
        super().__init__(
            tokenize_fnc=tokenizer,
            min_token_len=min_token_len,
            min_ctx_len=min_ctx_len,
            xml_aware=False,
        )
