import pytest

from livekit.agents.voice.transcription.filters import (
    apply_text_transforms,
    filter_emoji,
    filter_markdown,
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


# Complex TTS preprocessing test combining multiple filters
TTS_INPUT = """Meeting on 2024-12-25 with 15% discount!
Call 555-123-4567 or email john.doe@example.com for details.

**Special Offer**: $42.50 regular price, now only $12.99!
Distance: 15.3 mi from downtown. Weight: 85.6 kg capacity.

The API uses NASA technology with 99.5% uptime.
Population: 1,234,567 people in 2024.

```python
def calculate(x):
    return x * 2
```

Contact info:
- Phone: 555-987-6543
- Rate: 67%
- Price: $0.50/unit

Meeting at 14:30 today. Temperature: 72 degrees.
_Important_: This costs $5 and takes 3 hours.
I have 5 apples and 23 oranges.

HTML & CSS files on AWS S3 — $0.023/GB cost."""

TTS_EXPECTED_OUTPUT = """Meeting on Wednesday, December twenty five, 2024 with fifteen percent discount!
Call five five five one two three four five six seven or email john dot doe at example dot com for details.

**Special Offer**: forty two dollars and fifty cents regular price, now only twelve dollars and ninety nine cents!
Distance: 15 point 3 miles from downtown. Weight: 85 point 6 kilograms capacity.

The api uses nasa technology with 99 point 5 percent uptime.
Population: 1234567 people in 2024.


def calculate(x):
    return x * two


Contact info:
Phone: five five five nine eight seven six five four three
Rate: sixty seven percent
Price: zero dollars and fifty cents/unit

Meeting at 14:30 today. Temperature: seventy two degrees.
Important: This costs five dollars and takes three hours.
I have five apples and twenty three oranges.

html & css files on aws S3 — zero point zero two three dollars/G B cost."""


@pytest.mark.parametrize("chunk_size", [15, 50, 100])
async def test_tts_preprocessing_combined(chunk_size: int):
    """Comprehensive TTS preprocessing test combining multiple filters."""

    print("=== COMPREHENSIVE TTS PREPROCESSING TEST ===")
    print(f"Input length: {len(TTS_INPUT)} characters")
    print(f"Expected length: {len(TTS_EXPECTED_OUTPUT)} characters")

    print(f"\n--- Testing with chunk_size={chunk_size} ---")

    # Define the filter chain for TTS preprocessing (subset of DEFAULT_TTS_TEXT_TRANSFORMS)
    # Note: exclude replace_newlines_with_periods to preserve line structure for testing
    filters = [
        "filter_markdown",
        "filter_emoji",
        "format_dates",
        "format_times",
        "format_emails",
        "format_phone_numbers",
        "format_acronyms",
        "format_dollar_amounts",
        "format_distances",
        "format_units",
        "format_percentages",
        "format_numbers",  # Must be LAST to not interfere with other formatters
    ]

    # Stream the input with specified chunk size
    async def stream_text():
        for i in range(0, len(TTS_INPUT), chunk_size):
            yield TTS_INPUT[i : i + chunk_size]

    # Process through the filter chain
    result = ""
    async for chunk in apply_text_transforms(stream_text(), filters):
        result += chunk

    # Compare results
    if result.strip() == TTS_EXPECTED_OUTPUT.strip():
        print("✓ PASS")
    else:
        print("✗ FAIL")
        print(f"Expected first 150 chars: {repr(TTS_EXPECTED_OUTPUT[:150])}")
        print(f"Got first 150 chars:      {repr(result[:150])}")

        # Show differences
        expected_lines = TTS_EXPECTED_OUTPUT.strip().split("\n")
        result_lines = result.strip().split("\n")

        print("\nLine-by-line differences:")
        max_lines = max(len(expected_lines), len(result_lines))
        for i in range(max_lines):
            exp = expected_lines[i] if i < len(expected_lines) else "<missing>"
            got = result_lines[i] if i < len(result_lines) else "<missing>"
            if exp != got:
                print(f"Line {i + 1}:")
                print(f"  Expected: {repr(exp)}")
                print(f"  Got:      {repr(got)}")

    assert result.strip() == TTS_EXPECTED_OUTPUT.strip()

    print("\n=== TTS PREPROCESSING TEST COMPLETE ===")
