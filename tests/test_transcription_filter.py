import pytest

from livekit.agents.voice.transcription.filters import filter_markdown

MARKDOWN_INPUT = """# Mathematics and Markdown Guide

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

Regular text continues with normal punctuation! Math symbols like + - * / are preserved when not at line start."""  # noqa: E501


MARKDOWN_EXPECTED_OUTPUT = """Mathematics and Markdown Guide

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


Regular text continues with normal punctuation! Math symbols like + - * / are preserved when not at line start."""  # noqa: E501


@pytest.mark.parametrize("chunk_size", [1, 5, 10, 50])
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
        for i, (exp, got) in enumerate(zip(expected_lines, result_lines)):
            if exp != got:
                print(f"Line {i + 1}:")
                print(f"  Expected: {repr(exp)}")
                print(f"  Got:      {repr(got)}")

    print("\n=== TEST COMPLETE ===")
