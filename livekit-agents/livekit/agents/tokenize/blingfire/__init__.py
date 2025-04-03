import inspect
import os
import os.path
import platform
from ctypes import byref, c_char_p, c_int, c_int32, cdll, create_string_buffer

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))


def load_dll():
    match platform.system(), platform.machine().lower():
        case "Windows", "arm64" | "aarch64":
            return cdll.LoadLibrary(
                os.path.join(path, "runtimes", "win-arm64", "native", "blingfiretokdll.dll")
            )
        case "Windows", "x86_64" | "amd64" | "x64":
            return cdll.LoadLibrary(
                os.path.join(path, "runtimes", "win-x64", "native", "blingfiretokdll.dll")
            )
        case "Darwin", "arm64":
            return cdll.LoadLibrary(
                os.path.join(path, "runtimes", "osx-arm64", "native", "libblingfiretokdll.dylib")
            )
        case "Darwin", "x86_64" | "amd64" | "x64":
            return cdll.LoadLibrary(
                os.path.join(path, "runtimes", "osx-x64", "native", "libblingfiretokdll.dylib")
            )
        case "Linux", "arm64" | "aarch64":
            return cdll.LoadLibrary(
                os.path.join(path, "runtimes", "linux-arm64", "native", "libblingfiretokdll.so")
            )
        case "Linux", "x86_64" | "amd64" | "x64":
            return cdll.LoadLibrary(
                os.path.join(path, "runtimes", "linux-x64", "native", "libblingfiretokdll.so")
            )


blingfire = load_dll()


def text_to_sentences(s):
    # get the UTF-8 bytes
    s_bytes = s.encode("utf-8")

    # allocate the output buffer
    o_bytes = create_string_buffer(len(s_bytes) * 2)
    o_bytes_count = len(o_bytes)

    # identify paragraphs
    o_len = blingfire.TextToSentences(
        c_char_p(s_bytes), c_int(len(s_bytes)), byref(o_bytes), c_int(o_bytes_count)
    )

    # check if no error has happened
    if -1 == o_len or o_len > o_bytes_count:
        return ""

    # compute the unicode string from the UTF-8 bytes
    return o_bytes.value.decode("utf-8")


def get_blingfiretok_version():
    return blingfire.GetBlingFireTokVersion()


def text_to_token_with_offsets(s, text_to_token_f, split_byte):
    # get the UTF-8 bytes
    s_bytes = s.encode("utf-8")

    # allocate the output buffer
    o_bytes = create_string_buffer(len(s_bytes) * 2)
    o_bytes_count = len(o_bytes)

    # buffers for word beging and end
    o_start_offsets = (c_int32 * o_bytes_count)()
    o_end_offsets = (c_int32 * o_bytes_count)()

    # identify paragraphs
    o_len = text_to_token_f(
        c_char_p(s_bytes),
        c_int(len(s_bytes)),
        byref(o_bytes),
        byref(o_start_offsets),
        byref(o_end_offsets),
        c_int(o_bytes_count),
    )

    # check if no error has happened
    if -1 == o_len or o_len > o_bytes_count:
        return "", []

    num_tokens = o_bytes.value.count(split_byte) + 1

    utf8_offsets = [
        o
        for start_end in zip(o_start_offsets[:num_tokens], o_end_offsets[:num_tokens])
        for o in start_end
    ]

    string_offsets = []

    # Map the utf8 offsets to offsets in the original string - will break for "extended grapheme"
    # This seems to be undoing the FAUtf8Size based mapping being done inside TextToWordsWithOffsets and
    # TextToSentencesWithOffsets.
    string_offset = 0
    is_end_offset = False
    for utf8_offset, b in enumerate(s_bytes):
        if b & 0xC0 != 0x80:
            while (
                len(string_offsets) < len(utf8_offsets)
                and utf8_offsets[len(string_offsets)] + is_end_offset == utf8_offset
            ):
                string_offsets.append(string_offset)
                is_end_offset = not is_end_offset
            string_offset += 1

    if len(string_offsets) < num_tokens * 2:
        string_offsets.append(len(s))

    assert len(string_offsets) == num_tokens * 2, "%s != %s" % (len(string_offsets), num_tokens * 2)

    token_begin_end = [(b, e) for b, e in zip(string_offsets[::2], string_offsets[1::2])]

    # compute the unicode string from the UTF-8 bytes
    out_string = o_bytes.value.decode("utf8")

    return out_string, token_begin_end


def text_to_sentences_and_offsets(s):
    return text_to_token_with_offsets(s, blingfire.TextToSentencesWithOffsets, ord("\n"))
