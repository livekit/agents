import pytest

from livekit.agents.voice.transcription.filters import filter_emoji, filter_markdown
from livekit.agents.voice.transcription.text_transforms import _apply_text_transforms, replace

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]


async def _stream_text(text: str, chunk_size: int):
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


async def _collect(stream) -> str:
    result = ""
    async for chunk in stream:
        result += chunk
    return result


async def _collect_chunks(stream) -> list[str]:
    return [chunk async for chunk in stream]


async def _filtered(text: str, chunk_size: int) -> str:
    return await _collect(filter_markdown(_stream_text(text, chunk_size)))


MARKDOWN_INPUT = """# Mathematics and Markdown Guide

Hi there~~ How are you?  # the ~~ shouldn't be removed.
This document shows **bold text** and *italic text* with some math.

## Basic Math Operations
- Addition: 2 + 3 = 5
- Subtraction: 10 - 4 = 6
- Multiplication: 3 * 7 = 21
- Division: 15 / 3 = 5
- Comparison: 10 > 5 is true

### Code Examples
Use `print()` function to display: `print(2 + 3 * 4)`

The result is **14** because multiplication has higher precedence.

> Important: Order of operations matters in math!
> Remember: PEMDAS (Parentheses, Exponents, Multiplication/Division, Addition/Subtraction)

## Advanced Examples
1. Complex equation: (a + b) * c = result
2. Variable assignment: x = 5, y = 3
3. Conditional: if x > y then...

Here's a [useful calculator](https://calculator.com) for verification.

![Math diagram](diagram.png) shows the relationships.

~~Outdated formula~~ has been removed.

```python
def calculate(a, b):
    return a + b * 2
```

Regular text continues with normal punctuation! Math symbols like + - * / are preserved when not at line start.

## Identifier and Variable Tests
Here we test variables like test_case_one, my_variable_name, and function_names_with_underscores.
Private variables like _private_var and __dunder_method__ should remain unchanged.
Constants like MAX_SIZE_LIMIT and file_path_example.py are common.

But _this should be italic_ and **this should be bold** in markdown.
Also *single asterisk italic* and __double underscore bold__ should work.

Code identifiers: calculate_total(), get_user_data(), process_file_contents()
Class names: MyClass_Name, some_module_function, API_ENDPOINT_URL

This is a sentence. 这是一个中文句子。これは日本語の文章です。你好！LiveKit是一个直播音频和视频应用程序和服务的平台，我们正在测试Markdown~~过滤~~。"""  # noqa: E501


MARKDOWN_EXPECTED_OUTPUT = """Mathematics and Markdown Guide

Hi there~~ How are you?  # the ~~ shouldn't be removed.
This document shows bold text and italic text with some math.

Basic Math Operations
Addition: 2 + 3 = 5
Subtraction: 10 - 4 = 6
Multiplication: 3 * 7 = 21
Division: 15 / 3 = 5
Comparison: 10 > 5 is true

Code Examples
Use print() function to display: print(2 + 3 * 4)

The result is 14 because multiplication has higher precedence.

Important: Order of operations matters in math!
Remember: PEMDAS (Parentheses, Exponents, Multiplication/Division, Addition/Subtraction)

Advanced Examples
1. Complex equation: (a + b) * c = result
2. Variable assignment: x = 5, y = 3
3. Conditional: if x > y then...

Here's a useful calculator for verification.

Math diagram shows the relationships.

 has been removed.


def calculate(a, b):
    return a + b * 2


Regular text continues with normal punctuation! Math symbols like + - * / are preserved when not at line start.

Identifier and Variable Tests
Here we test variables like test_case_one, my_variable_name, and function_names_with_underscores.
Private variables like _private_var and __dunder_method__ should remain unchanged.
Constants like MAX_SIZE_LIMIT and file_path_example.py are common.

But this should be italic and this should be bold in markdown.
Also single asterisk italic and double underscore bold should work.

Code identifiers: calculate_total(), get_user_data(), process_file_contents()
Class names: MyClass_Name, some_module_function, API_ENDPOINT_URL

This is a sentence. 这是一个中文句子。これは日本語の文章です。你好！LiveKit是一个直播音频和视频应用程序和服务的平台，我们正在测试Markdown。"""  # noqa: E501


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 11, 50])
async def test_markdown_filter(chunk_size: int):
    """Mixed markdown, math and prose survive the streaming filter intact."""
    assert await _filtered(MARKDOWN_INPUT, chunk_size) == MARKDOWN_EXPECTED_OUTPUT.strip()


# --- emphasis that must be stripped ---
#
# One table per direction. This one covers every emphasis form the filter is
# expected to remove: the three delimiter widths, boundaries made of punctuation
# rather than whitespace, and delimiters flush against a word character.

EMPHASIS_CASES = [
    # (input, expected)
    # single and double, whitespace boundaries
    ("He said *hello* again.", "He said hello again."),
    ("This is **bold** text.", "This is bold text."),
    ("_underscore italic_ here.", "underscore italic here."),
    ("__underscore bold__ here.", "underscore bold here."),
    # triple: bold italic
    ("This is ***very important*** text.", "This is very important text."),
    ("Call ***now***!", "Call now!"),
    ("___Totally___ critical.", "Totally critical."),
    ("Use ***both*** and **bold** and *italic*.", "Use both and bold and italic."),
    # punctuation rather than whitespace on the boundary
    ("Hi **Frankie**!", "Hi Frankie!"),
    ("See **Dr. Smith**.", "See Dr. Smith."),
    ("Scheduled for **Monday**, at 9am.", "Scheduled for Monday, at 9am."),
    ("Press (**1**) to confirm.", "Press (1) to confirm."),
    ('Try "**this**" first.', 'Try "this" first.'),
    ("Options: **a**, **b**, or **c**?", "Options: a, b, or c?"),
    ("He said *hello*!", "He said hello!"),
    ("It was *amazing*, really.", "It was amazing, really."),
    ("Hi ***Frankie***!", "Hi Frankie!"),
    ("Press (***1***) to confirm.", "Press (1) to confirm."),
    # nested delimiters, stripped by successive passes
    ("**_mixed_** here.", "mixed here."),
    ("_**mixed**_ here.", "mixed here."),
    # scripts written without spaces
    ("这是**很重要**的文本。", "这是很重要的文本。"),
    ("这是***非常重要***的文本。", "这是非常重要的文本。"),
    ("这是*重要*的文本。", "这是重要的文本。"),
    ("テスト**強調**です。", "テスト強調です。"),
    ("**中文**开头。", "中文开头。"),
    ("นี่คือ**ข้อความ**สำคัญ", "นี่คือข้อความสำคัญ"),
    # korean: spaced between words, but a particle follows the closing run
    ("이것은 **중요**합니다.", "이것은 중요합니다."),
    ("한국어**강조**입니다.", "한국어강조입니다."),
    ("한국어***강조***입니다.", "한국어강조입니다."),
    ("이것은 *중요*합니다.", "이것은 중요합니다."),
]


@pytest.mark.parametrize("text,expected", EMPHASIS_CASES)
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 50])
async def test_emphasis_stripped(text: str, expected: str, chunk_size: int):
    assert await _filtered(text, chunk_size) == expected


# --- text that must survive untouched ---
#
# Asterisks and underscores carry meaning outside markdown: arithmetic, globs,
# exponentiation and identifiers. Delimiter runs that cannot pair up are left
# alone rather than half-stripped.

PRESERVE_CASES = [
    # arithmetic, globs, exponentiation
    "2 * 3 = 6",
    "Use *.py files",
    "x**2 + y**2 = z**2",
    "a ** b evaluated right-to-left",
    "cost = a *** b",
    # identifiers: underscores never emphasise intra-word
    "__dunder_method__ stays",
    "a___b and MAX___VALUE",
    "snake___case___name",
    "call some_function_name here",
    # CommonMark forbids intra-word underscore emphasis, so CJK keeps its
    # underscores even though CJK asterisk emphasis is stripped
    "テスト__強調__です。",
    "变量__name__的值",
    "한국어__강조__입니다.",
    # runs that cannot pair up
    "****quad****",
    "____quad____",
    "**bold***italic*",
    "*italic***bold**",
    # a run of asterisks is only a rule on a line of its own
    "see ***** here",
    # unterminated at end of stream
    "unterminated ***open",
    "unterminated ___open",
]


@pytest.mark.parametrize("text", PRESERVE_CASES)
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 50])
async def test_non_markdown_preserved(text: str, chunk_size: int):
    assert await _filtered(text, chunk_size) == text


# --- horizontal rules ---
#
# A rule carries no spoken content, so the whole line goes. Dashes that are
# part of a sentence must survive.

HORIZONTAL_RULE_CASES = [
    ("before\n---\nafter", "before\n\nafter"),
    ("before\n***\nafter", "before\n\nafter"),
    ("before\n___\nafter", "before\n\nafter"),
    ("before\n-----\nafter", "before\n\nafter"),
    ("before\n  ---  \nafter", "before\n\nafter"),
    ("*****", ""),
    # markers may be spaced apart, and a rule wins over a list item
    ("before\n* * *\nafter", "before\n\nafter"),
    ("before\n- - -\nafter", "before\n\nafter"),
    ("before\n_ _ _\nafter", "before\n\nafter"),
    ("before\n   - - - \nafter", "before\n\nafter"),
    # not rules
    ("wait --- what?", "wait --- what?"),
    ("a -- b", "a -- b"),
    # a fourth column of indent makes the line code, not a rule
    ("before\n    ---\nafter", "before\n    ---\nafter"),
    ("before\n\t---\nafter", "before\n\t---\nafter"),
]


@pytest.mark.parametrize("text,expected", HORIZONTAL_RULE_CASES)
@pytest.mark.parametrize("chunk_size", [1, 3, 50])
async def test_horizontal_rule(text: str, expected: str, chunk_size: int):
    assert await _filtered(text, chunk_size) == expected


# --- streaming invariant ---


@pytest.mark.parametrize(
    "text",
    [t for t, _ in EMPHASIS_CASES] + PRESERVE_CASES + [t for t, _ in HORIZONTAL_RULE_CASES],
)
async def test_output_independent_of_chunking(text: str):
    """Buffering must not change the result: every chunk size yields one output."""
    outputs = {await _filtered(text, size) for size in (1, 2, 3, 5, 7, 11, 50, 1000)}
    assert len(outputs) == 1, f"chunk-size dependent output: {outputs}"


# Emoji test data
EMOJI_INPUT = """Hello! 😀 Welcome to our app! 🎉

This message contains various emojis:
- Happy faces: 😊 😃 🙂 😄
- Hearts: ❤️ 💙 💚 💛 🧡 💜
- Animals: 🐶 🐱 🐸 🦊 🐘
- Food: 🍎 🍕 🍔 🍦 🎂
- Activities: ⚽ 🏀 🎮 🎵 📚
- Weather: ☀️ 🌙 ⭐ 🌈 ⛅
- Flags: 🇺🇸 🇬🇧 🇯🇵 🇩🇪

Complex emojis with modifiers:
- Skin tones: 👋🏻 👋🏽 👋🏿
- Gender variants: 👨‍💻 👩‍💻 🧑‍💻
- Family emojis: 👨‍👩‍👧‍👦 👩‍👩‍👧
- Professional: 👨‍⚕️ 👩‍🏫 👮‍♂️

Numbers with keycaps: 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣

Mixed content with regular text and punctuation! 🚀
The app works great. Let's celebrate! 🎊

End of emoji test. 🔚"""

EMOJI_EXPECTED_OUTPUT = """Hello!  Welcome to our app! 

This message contains various emojis:
- Happy faces:    
- Hearts:      
- Animals:     
- Food:     
- Activities:     
- Weather:     
- Flags:    

Complex emojis with modifiers:
- Skin tones:   
- Gender variants:   
- Family emojis:  
- Professional:   

Numbers with keycaps: 1 2 3 4 5

Mixed content with regular text and punctuation! 
The app works great. Let's celebrate! 

End of emoji test. """  # noqa: W291


@pytest.mark.parametrize("chunk_size", [1, 5, 10, 30])
async def test_emoji_filter(chunk_size: int):
    result = await _collect(filter_emoji(_stream_text(EMOJI_INPUT, chunk_size)))
    assert result == EMOJI_EXPECTED_OUTPUT


# --- text_transforms.replace tests ---


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 11, 50])
async def test_replace_across_chunk_sizes(chunk_size: int):
    """Test replacement with multiple keys, case insensitivity, and boundary spanning."""
    transform = replace({"LiveKit": "Lyve Kit", "SQL": "sequel", "boundary": "EDGE"})
    text = "LiveKit uses SQL. livekit boundary test."
    result = await _collect(transform(_stream_text(text, chunk_size)))
    assert result == "Lyve Kit uses sequel. Lyve Kit EDGE test."


@pytest.mark.parametrize("chunk_size", [1, 3, 7])
async def test_replace_case_sensitive(chunk_size: int):
    """Test case-sensitive mode only replaces exact matches."""
    transform = replace({"LiveKit": "Lyve Kit"}, case_sensitive=True)
    result = await _collect(
        transform(_stream_text("LiveKit is great. livekit should stay.", chunk_size))
    )
    assert result == "Lyve Kit is great. livekit should stay."


async def test_replace_edge_cases():
    """Test empty replacements, empty input, no matches, and regex special chars."""
    # empty replacements = passthrough
    text = "Hello world."
    assert await _collect(replace({})(_stream_text(text, 3))) == text

    # empty input
    assert await _collect(replace({"foo": "bar"})(_stream_text("", 1))) == ""

    # no matches
    assert await _collect(replace({"xyz": "abc"})(_stream_text(text, 4))) == text

    # regex special characters in keys
    transform = replace({"C++": "cpp", "file.txt": "file_txt"})
    assert (
        await _collect(transform(_stream_text("Use C++ to read file.txt", 2)))
        == "Use cpp to read file_txt"
    )

    # backslashes in replacement values are treated literally
    transform = replace({"word": r"\1 \n \t"})
    assert await _collect(transform(_stream_text("a word here", 2))) == r"a \1 \n \t here"


async def test_replace_flushes_non_prefix_text_immediately():
    """Text that cannot begin any key is emitted whole, not held or split mid-word."""
    transform = replace({"LiveKit": "Lyve Kit"})
    chunks = await _collect_chunks(transform(_stream_text("you connect.", 100)))
    assert chunks == ["you connect."]


async def test_replace_holds_only_potential_prefix():
    """A trailing run that is a prefix of a key is held until it can be resolved."""
    transform = replace({"LiveKit": "Lyve Kit"})
    chunks = await _collect_chunks(transform(_stream_text("visit Live", 100)))
    # "visit " flushes immediately; only "Live" (a LiveKit prefix) is held to the end
    assert chunks[0] == "visit "
    assert "".join(chunks) == "visit Live"


async def test_replace_prefers_longest_overlapping_key():
    """Overlapping keys resolve to the longest match, not dict/insertion order.

    Longest-match holds when the longer key is buffered together; a key split
    mid-token across chunks falls back to the shorter match (same as before —
    correcting that needs incremental leftmost-longest matching).
    """
    transform = replace({"a": "X", "ab": "Y"})
    assert await _collect(transform(_stream_text("ab", 100))) == "Y"


async def test_replace_does_not_cascade():
    """A replacement's output is not re-matched against other keys (single pass)."""
    transform = replace({"a": "b", "b": "c"})
    assert await _collect(transform(_stream_text("a", 100))) == "b"


async def test_apply_text_transforms_with_callable():
    """Test _apply_text_transforms with callable, mixed transforms, and invalid input."""
    # callable only
    result = await _collect(
        _apply_text_transforms(_stream_text("Hello world!", 3), [replace({"world": "planet"})])
    )
    assert result == "Hello planet!"

    # mixed: builtin string + callable + builtin string
    result = await _collect(
        _apply_text_transforms(
            _stream_text("**hello** world! 😀", 3),
            ["filter_markdown", replace({"hello": "hi"}), "filter_emoji"],
        )
    )
    assert result == "hi world! "

    # invalid string transform
    with pytest.raises(ValueError, match="Invalid transform"):
        await _collect(_apply_text_transforms(_stream_text("text", 4), ["nonexistent"]))
