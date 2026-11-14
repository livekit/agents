import pytest

from livekit.agents.voice.transcription.filters import filter_emoji, filter_markdown
from livekit.agents.voice.transcription.text_transforms import _apply_text_transforms, replace

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]

# every streamed result must be the same whatever the chunk boundaries
CHUNK_SIZES = (1, 2, 3, 5, 7, 11, 50, 1000)


async def _stream_text(text: str, chunk_size: int):
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


async def _collect(stream) -> str:
    return "".join([chunk async for chunk in stream])


async def _collect_chunks(stream) -> list[str]:
    return [chunk async for chunk in stream]


async def _outputs(transform, text: str) -> set[str]:
    return {await _collect(transform(_stream_text(text, size))) for size in CHUNK_SIZES}


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


async def test_markdown_filter():
    """Mixed markdown, math and prose survive the streaming filter intact."""
    assert await _outputs(filter_markdown, MARKDOWN_INPUT) == {MARKDOWN_EXPECTED_OUTPUT.strip()}


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
async def test_emphasis_stripped(text: str, expected: str):
    assert await _outputs(filter_markdown, text) == {expected}


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


# --- delimiters that can never pair up must not hold the stream ---

STRAY_DELIMITER_CASES = [
    "the confirmation went to first_last@example.com and your table is ready.",
    "10*4 is 40 and that is four tens.",
    "10 * 4 is 40 and that is four tens.",
    "rates rose last year [1] and may rise again.",
    "see ]( broken formatting and please continue.",
]


@pytest.mark.parametrize("text", PRESERVE_CASES + STRAY_DELIMITER_CASES)
async def test_non_markdown_preserved(text: str):
    assert await _outputs(filter_markdown, text) == {text}


@pytest.mark.parametrize("text", STRAY_DELIMITER_CASES)
async def test_stray_delimiter_does_not_hold_the_stream(text: str):
    words = text.split(" ")
    consumed = 0

    async def word_stream():
        nonlocal consumed
        for word in words:
            consumed += 1
            yield word + " "

    output = ""
    consumed_at_and = None
    async for chunk in filter_markdown(word_stream()):
        output += chunk
        if consumed_at_and is None and " and" in output:
            consumed_at_and = consumed

    assert output == text + " "
    assert consumed_at_and is not None and consumed_at_and < len(words)


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
async def test_horizontal_rule(text: str, expected: str):
    assert await _outputs(filter_markdown, text) == {expected}


LINK_CASES = [
    ("see [this](https://example.com) now", "see this now"),
    # a destination may hold balanced parentheses, as Wikipedia URLs often do
    ("read [Python](https://en.wikipedia.org/wiki/Python_(language)) first", "read Python first"),
    ("![the logo](img_(1).png) here", "the logo here"),
    # a backslash-escaped paren opens no group
    (r"read [guide](https://example.com/a\(b) now", "read guide now"),
    (r"![logo](img\(1.png) here", "logo here"),
]


@pytest.mark.parametrize("text,expected", LINK_CASES)
async def test_link_text_kept(text: str, expected: str):
    assert await _outputs(filter_markdown, text) == {expected}


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


async def test_emoji_filter():
    assert await _outputs(filter_emoji, EMOJI_INPUT) == {EMOJI_EXPECTED_OUTPUT}


# --- text_transforms.replace ---


@pytest.mark.parametrize(
    "replacements, text, expected",
    [
        # case-insensitive by default, several keys
        (
            {"LiveKit": "Lyve Kit", "SQL": "sequel", "boundary": "EDGE"},
            "LiveKit uses SQL. livekit boundary test.",
            "Lyve Kit uses sequel. Lyve Kit EDGE test.",
        ),
        ({"LiveKit": "Lyve Kit"}, "a LiveKit and livekit agent", "a Lyve Kit and Lyve Kit agent"),
        ({"um": ""}, "um well um", " well "),
        ({"ab": "cab"}, "xabc ab", "xcabc cab"),
        # overlapping keys resolve to the longest match, not insertion order
        ({"a": "X", "ab": "Y"}, "ab", "Y"),
        # a replacement's output is never re-matched (single pass)
        ({"a": "b", "b": "c"}, "a", "b"),
        ({"hello": "hi", "hi there": "GT"}, "hello there, hello there", "hi there, hi there"),
        # input right after a replacement is not swallowed by the held-back output
        ({"ab": "a"}, "abb", "ab"),
        ({"Texas": "Tex"}, "Texasas", "Texas"),
    ],
)
async def test_replace(replacements: dict[str, str], text: str, expected: str):
    assert await _outputs(replace(replacements), text) == {expected}


async def test_replace_case_sensitive():
    transform = replace({"LiveKit": "Lyve Kit"}, case_sensitive=True)
    assert await _outputs(transform, "LiveKit is great. livekit should stay.") == {
        "Lyve Kit is great. livekit should stay."
    }


async def test_replace_edge_cases():
    """Empty maps, empty input, no matches and regex metacharacters pass through literally."""
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


async def test_apply_text_transforms_with_callable():
    """Callables and builtin names chain in order; an unknown name raises."""
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
