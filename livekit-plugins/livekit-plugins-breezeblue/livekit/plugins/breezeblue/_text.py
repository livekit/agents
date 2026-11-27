from collections.abc import AsyncIterable, AsyncIterator, Iterator

from livekit.agents import tokenize, tts


def take_prefix(text: str, limit: int) -> tuple[str, str]:
    """Prefer a whitespace boundary while retaining every input character."""
    if len(text) <= limit:
        return text, ""
    boundary = max((i for i in range(limit // 2, limit) if text[i].isspace()), default=-1)
    end = boundary + 1 if boundary >= 0 else limit
    return text[:end], text[end:]


def utf8_chunks(text: str, limit: int) -> Iterator[str]:
    chunk = ""
    size = 0
    for char in text:
        width = len(char.encode("utf-8"))
        if size + width > limit:
            yield chunk
            chunk, size = "", 0
        chunk += char
        size += width
    if chunk:
        yield chunk


async def text_segments(
    input_channel: AsyncIterable[str | tts.SynthesizeStream._FlushSentinel],
    limit: int,
    *,
    whole_response: bool,
    input_chunk_length_schedule: tuple[int, ...] | None = None,
) -> AsyncIterator[str | None]:
    if whole_response and input_chunk_length_schedule is None:
        text = "".join([part async for part in input_channel if isinstance(part, str)])
        if text.strip():
            yield text
        return

    tokenizer = (
        tokenize.blingfire.SentenceTokenizer(min_sentence_len=1, retain_format=True)
        if input_chunk_length_schedule is None
        else None
    )
    chunk_index = 0
    buffer = ""
    async for data in input_channel:
        if not isinstance(data, str):
            if buffer.strip():
                yield buffer
            buffer = ""
            chunk_index = 0
            yield None
            continue
        buffer += data
        while buffer.strip():
            if input_chunk_length_schedule is not None:
                length = input_chunk_length_schedule[chunk_index]
                if len(buffer) < length:
                    break
                text, buffer = buffer[:length], buffer[length:]
                if text.strip():
                    yield text
                    chunk_index = min(chunk_index + 1, len(input_chunk_length_schedule) - 1)
                continue
            assert tokenizer is not None
            sentences = tokenizer.tokenize(buffer)
            if len(sentences) > 1:
                split = buffer.index(sentences[0]) + len(sentences[0])
                text, buffer = buffer[:split], buffer[split:]
            elif len(buffer) >= limit:
                text, buffer = take_prefix(buffer, limit)
            else:
                break
            yield text
    if buffer.strip():
        yield buffer
