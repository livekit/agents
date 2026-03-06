import pytest

from livekit.agents.voice.transcription.filters import (
    filter_emoji,
    filter_markdown,
    filter_ssml,
    strip_ssml,
)

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
    """Comprehensive test with mixed markdown, math operations, and regular text."""

    print("=== COMPREHENSIVE MARKDOWN FILTER TEST ===")
    print(f"Input length: {len(MARKDOWN_INPUT)} characters")
    print(f"Expected length: {len(MARKDOWN_EXPECTED_OUTPUT)} characters")

    print(f"\n--- Testing with chunk_size={chunk_size} ---")

    # Stream the input with specified chunk size
    async def stream_text():
        for i in range(0, len(MARKDOWN_INPUT), chunk_size):
            yield MARKDOWN_INPUT[i : i + chunk_size]

    # Process through the filter
    result = ""
    async for chunk in filter_markdown(stream_text()):
        result += chunk

    # Compare results
    if result.strip() == MARKDOWN_EXPECTED_OUTPUT.strip():
        print("✓ PASS")
    else:
        print("✗ FAIL")
        print(f"Expected first 100 chars: {repr(MARKDOWN_EXPECTED_OUTPUT[:100])}")
        print(f"Got first 100 chars:      {repr(result[:100])}")

        # Show differences
        expected_lines = MARKDOWN_EXPECTED_OUTPUT.strip().split("\n")
        result_lines = result.strip().split("\n")

        print("\nLine-by-line differences:")
        for i, (exp, got) in enumerate(zip(expected_lines, result_lines, strict=False)):
            if exp != got:
                print(f"Line {i + 1}:")
                print(f"  Expected: {repr(exp)}")
                print(f"  Got:      {repr(got)}")
    assert result == MARKDOWN_EXPECTED_OUTPUT.strip()

    print("\n=== TEST COMPLETE ===")


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
    """Test emoji filtering with various chunk sizes."""

    print("=== EMOJI FILTER TEST ===")
    print(f"Input length: {len(EMOJI_INPUT)} characters")
    print(f"Expected length: {len(EMOJI_EXPECTED_OUTPUT)} characters")

    print(f"\n--- Testing with chunk_size={chunk_size} ---")

    # Stream the input with specified chunk size
    async def stream_text():
        for i in range(0, len(EMOJI_INPUT), chunk_size):
            yield EMOJI_INPUT[i : i + chunk_size]

    # Process through the filter
    result = ""
    async for chunk in filter_emoji(stream_text()):
        result += chunk

    # Compare results
    if result == EMOJI_EXPECTED_OUTPUT:
        print("✓ PASS")
    else:
        print("✗ FAIL")
        print(f"Expected first 100 chars: {repr(EMOJI_EXPECTED_OUTPUT[:100])}")
        print(f"Got first 100 chars:      {repr(result[:100])}")

        # Show differences
        expected_lines = EMOJI_EXPECTED_OUTPUT.split("\n")
        result_lines = result.split("\n")

        print("\nLine-by-line differences:")
        for i, (exp, got) in enumerate(zip(expected_lines, result_lines, strict=False)):
            if exp != got:
                print(f"Line {i + 1}:")
                print(f"  Expected: {repr(exp)}")
                print(f"  Got:      {repr(got)}")
    assert result == EMOJI_EXPECTED_OUTPUT

    print("\n=== EMOJI TEST COMPLETE ===")


# SSML test data
SSML_INPUT = (
    'Hello, welcome to our service.<break time="1s"/>'
    "We are glad to have you here."
    '<break time="500ms" />'
    "<speak>This is inside speak tags.</speak>"
    ' And <prosody rate="slow">this is slow speech</prosody> done.'
)

SSML_EXPECTED_OUTPUT = (
    "Hello, welcome to our service."
    "We are glad to have you here."
    "This is inside speak tags."
    " And this is slow speech done."
)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 11, 50, 200])
async def test_ssml_filter(chunk_size: int):
    """Test SSML tag filtering with various chunk sizes."""

    async def stream_text():
        for i in range(0, len(SSML_INPUT), chunk_size):
            yield SSML_INPUT[i : i + chunk_size]

    result = ""
    async for chunk in filter_ssml(stream_text()):
        result += chunk

    assert result == SSML_EXPECTED_OUTPUT, (
        f"chunk_size={chunk_size}: expected {SSML_EXPECTED_OUTPUT!r}, got {result!r}"
    )


async def test_ssml_filter_no_tags():
    """Plain text without SSML tags should pass through unchanged."""

    plain = "Hello, this is a normal sentence without any tags."

    async def stream_text():
        yield plain

    result = ""
    async for chunk in filter_ssml(stream_text()):
        result += chunk

    assert result == plain


async def test_ssml_filter_only_tags():
    """Text consisting entirely of SSML tags should produce empty output."""

    async def stream_text():
        yield '<break time="1s"/><speak></speak>'

    result = ""
    async for chunk in filter_ssml(stream_text()):
        result += chunk

    assert result == ""


async def test_ssml_filter_self_closing_variants():
    """Test various self-closing SSML tag formats."""

    input_text = 'Hello<break time="1s"/>world<break time="500ms" /> end'
    expected = "Helloworld end"

    async def stream_text():
        yield input_text

    result = ""
    async for chunk in filter_ssml(stream_text()):
        result += chunk

    assert result == expected


def test_strip_ssml():
    """Test the synchronous strip_ssml helper."""
    assert strip_ssml('Hello <break time="1s"/> world') == "Hello  world"
    assert strip_ssml("<speak>text</speak>") == "text"
    assert strip_ssml("no tags here") == "no tags here"
    assert strip_ssml('<prosody rate="slow">slow</prosody>') == "slow"
    assert strip_ssml("") == ""
