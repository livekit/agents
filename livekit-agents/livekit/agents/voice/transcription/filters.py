import re
from collections.abc import AsyncIterable, Callable, Sequence
from typing import Literal, Optional, Union

# Number to word mappings for TTS preprocessing
ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEENS = [
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

# Known acronyms that should be lowercased for TTS
KNOWN_ACRONYMS = {
    # Organizations & Countries
    "NASA",
    "FBI",
    "CIA",
    "USA",
    "UK",
    "EU",
    "UN",
    "NATO",
    "WHO",
    # Web & Protocols
    "API",
    "REST",
    "SOAP",
    "URL",
    "URI",
    "HTTP",
    "HTTPS",
    "FTP",
    "SMTP",
    "SSH",
    "SSL",
    "TLS",
    "TCP",
    "UDP",
    "IP",
    "DNS",
    "CORS",
    "AJAX",
    "RPC",
    "GRPC",
    "MQTT",
    "AMQP",
    "RTMP",
    "RTSP",
    "SIP",
    # Languages & Markup
    "HTML",
    "CSS",
    "XML",
    "JSON",
    "YAML",
    "TOML",
    "SVG",
    "PHP",
    "SQL",
    "JSX",
    "TSX",
    # Databases
    "RDBMS",
    "OLAP",
    "OLTP",
    "ACID",
    "CRUD",
    "ORM",
    # Cloud & Services
    "AWS",
    "GCP",
    "SaaS",
    "PaaS",
    "IaaS",
    "CDN",
    "VPN",
    "S3",
    "EC2",
    "ECS",
    "EKS",
    "RDS",
    "IAM",
    "SDK",
    # DevOps & Tools
    "CI",
    "CD",
    "CICD",
    "CLI",
    "IDE",
    "GIT",
    "NPM",
    "YARN",
    "JWT",
    "OAuth",
    "SAML",
    "LDAP",
    # Architecture & Patterns
    "MVC",
    "MVVM",
    "MVP",
    "SPA",
    "SSR",
    "CSR",
    "SSG",
    "PWA",
    "WYSIWYG",
    "REPL",
    # AI & ML
    "AI",
    "ML",
    "NLP",
    "CNN",
    "RNN",
    "GAN",
    "LLM",
    "GPT",
    "GPU",
    "CPU",
    "TPU",
    # Technologies
    "IoT",
    "VR",
    "AR",
    "GPS",
    "NFC",
    "RFID",
    "QR",
    "OCR",
    "TTS",
    "ASR",
    # Standards & Specs
    "IEEE",
    "ISO",
    "ANSI",
    "RFC",
    "W3C",
    "ECMA",
    "POSIX",
    "UTF",
    "ASCII",
    "UUID",
    "GUID",
    "MIME",
    # Software Development
    "TDD",
    "BDD",
    "DDD",
    "SOLID",
    "DRY",
    "KISS",
    "YAGNI",
    "FIFO",
    "LIFO",
    "LGTM",
    # Performance & Monitoring
    "SLA",
    "SLO",
    "SLI",
    "APM",
    "RPS",
    "QPS",
    "TTL",
    "TTFB",
    "LCP",
    "FCP",
    "CLS",
}

# Common street abbreviations
DEFAULT_STREET_REPLACEMENTS = [
    (re.compile(r"\b([\w\s]+)\s+ST\b", re.IGNORECASE), r"\1 street"),
    (re.compile(r"\b([\w\s]+)\s+RD\b", re.IGNORECASE), r"\1 road"),
    (re.compile(r"\b([\w\s]+)\s+AVE\b", re.IGNORECASE), r"\1 avenue"),
    (re.compile(r"\b([\w\s]+)\s+BLVD\b", re.IGNORECASE), r"\1 boulevard"),
    (re.compile(r"\b([\w\s]+)\s+DR\b", re.IGNORECASE), r"\1 drive"),
    (re.compile(r"\b([\w\s]+)\s+LN\b", re.IGNORECASE), r"\1 lane"),
    (re.compile(r"\b([\w\s]+)\s+CT\b", re.IGNORECASE), r"\1 court"),
    (re.compile(r"\b([\w\s]+)\s+PL\b", re.IGNORECASE), r"\1 place"),
]


def number_to_words(num: int) -> str:
    """Convert a number (0-99) to words for TTS."""
    if num == 0:
        return "zero"
    if num < 10:
        return ONES[num] if num < len(ONES) else str(num)
    if num < 20:
        return TEENS[num - 10] if (num - 10) < len(TEENS) else str(num)
    if num < 100:
        ten = num // 10
        one = num % 10
        result = TENS[ten] if ten < len(TENS) else ""
        if one > 0:
            result += " " + (ONES[one] if one < len(ONES) else "")
        return result.strip()
    return str(num)


async def _buffered_regex_filter(
    text: AsyncIterable[str],
    pattern: Union[str, re.Pattern[str]],
    process_match: Callable[[re.Match[str]], str],
    preprocess_chunk: Optional[Callable[[str], str]] = None,
    flags: int = 0,
) -> AsyncIterable[str]:
    """
    Generic helper for buffered regex-based text filtering.

    Args:
        text: Input text stream
        pattern: Regex pattern to match
        process_match: Function that takes a match object and returns replacement string
        preprocess_chunk: Optional function to preprocess each chunk before pattern matching
        flags: Regex flags
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern, flags)

    buffer = ""

    async for chunk in text:
        buffer += chunk

        # Look for sentence-ending punctuation followed by space or newline
        last_safe_pos = 0
        for i in range(len(buffer) - 1):
            if buffer[i] in ".!?" and (buffer[i + 1] in " \n\t" or buffer[i + 1].isupper()):
                last_safe_pos = i + 1

        # Also check if buffer ends with newline
        if buffer and buffer[-1] == "\n":
            last_safe_pos = len(buffer)

        if last_safe_pos > 0:
            processable = buffer[:last_safe_pos]
            rest = buffer[last_safe_pos:]

            # Apply preprocessing if provided
            if preprocess_chunk:
                processable = preprocess_chunk(processable)

            # Process all matches with offset tracking
            result = processable
            offset = 0
            for match in pattern.finditer(processable):
                replacement = process_match(match)
                start_pos = match.start() + offset
                end_pos = match.end() + offset
                result = result[:start_pos] + replacement + result[end_pos:]
                offset += len(replacement) - len(match.group(0))

            yield result
            buffer = rest

    # Process remaining buffer
    if buffer:
        if preprocess_chunk:
            buffer = preprocess_chunk(buffer)

        result = buffer
        offset = 0
        for match in pattern.finditer(buffer):
            replacement = process_match(match)
            start_pos = match.start() + offset
            end_pos = match.end() + offset
            result = result[:start_pos] + replacement + result[end_pos:]
            offset += len(replacement) - len(match.group(0))
        yield result


TextTransforms = Literal[
    "filter_markdown",
    "filter_emoji",
    "format_numbers",
    "format_acronyms",
    "format_dollar_amounts",
    "format_emails",
    "format_dates",
    "format_times",
    "format_distances",
    "format_units",
    "format_percentages",
    "format_phone_numbers",
    "remove_angle_bracket_content",
    "replace_newlines_with_periods",
]


def apply_text_transforms(
    text: AsyncIterable[str], transforms: Sequence[TextTransforms]
) -> AsyncIterable[str]:
    all_transforms = {
        "filter_markdown": filter_markdown,
        "filter_emoji": filter_emoji,
        "format_numbers": format_numbers,
        "format_acronyms": format_acronyms,
        "format_dollar_amounts": format_dollar_amounts,
        "format_emails": format_emails,
        "format_dates": format_dates,
        "format_times": format_times,
        "format_distances": format_distances,
        "format_units": format_units,
        "format_percentages": format_percentages,
        "format_phone_numbers": format_phone_numbers,
        "remove_angle_bracket_content": remove_angle_bracket_content,
        "replace_newlines_with_periods": replace_newlines_with_periods,
    }

    for transform in transforms:
        if transform not in all_transforms:
            raise ValueError(
                f"Invalid transform: {transform}, available transforms: {all_transforms.keys()}"
            )
        text = all_transforms[transform](text)
    return text


LINE_PATTERNS = [
    # headers: remove # and following spaces
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    # list markers: remove -, +, * and following spaces
    (re.compile(r"^\s*[-+*]\s+", re.MULTILINE), ""),
    # block quotes: remove > and following spaces
    (re.compile(r"^\s*>\s+", re.MULTILINE), ""),
]

INLINE_PATTERNS = [
    # images: keep alt text ![alt](url) -> alt
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # links: keep text part [text](url) -> text
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # bold: remove asterisks from **text** (not preceded/followed by non-whitespace)
    (re.compile(r"(?<!\S)\*\*([^*]+?)\*\*(?!\S)"), r"\1"),
    # italic: remove asterisks from *text* (not preceded/followed by non-whitespace)
    (re.compile(r"(?<!\S)\*([^*]+?)\*(?!\S)"), r"\1"),
    # bold with underscores: remove underscores from __text__ (word boundaries)
    (re.compile(r"(?<!\w)__([^_]+?)__(?!\w)"), r"\1"),
    # italic with underscores: remove underscores from _text_ (word boundaries)
    (re.compile(r"(?<!\w)_([^_]+?)_(?!\w)"), r"\1"),
    # code blocks: remove ``` from ```text```
    (re.compile(r"`{3,4}[\S]*"), ""),
    # inline code: remove ` from `text`
    (re.compile(r"`([^`]+?)`"), r"\1"),
    # strikethrough: remove ~~text~~ (no spaces next to tildes)
    (re.compile(r"~~(?!\s)([^~]*?)(?<!\s)~~"), ""),
]
INLINE_SPLIT_TOKENS = " ,.?!;，。？！；"

COMPLETE_LINKS_PATTERN = re.compile(r"\[[^\]]*\]\([^)]*\)")  # links [text](url)
COMPLETE_IMAGES_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # images ![text](url)


async def filter_markdown(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Filter out markdown symbols from the text.
    """

    def has_incomplete_pattern(buffer: str) -> bool:
        """Check if buffer might contain incomplete markdown patterns that need more text."""

        if buffer.endswith(("#", "-", "+", "*", ">", "!", "`", "~", " ")):
            return True

        # check for incomplete bold (**text** or *text*)
        double_asterisks = buffer.count("**")
        if double_asterisks % 2 == 1:
            return True

        single_asterisks = buffer.count("*") - (double_asterisks * 2)
        if single_asterisks % 2 == 1:
            return True

        # check for incomplete underscores (__text__ or _text_)
        double_underscores = buffer.count("__")
        if double_underscores % 2 == 1:
            return True
        single_underscores = buffer.count("_") - (double_underscores * 2)
        if single_underscores % 2 == 1:
            return True

        # check for incomplete code (`text`)
        backticks = buffer.count("`")
        if backticks % 2 == 1:
            return True

        # check for incomplete strikethrough (~~text~~)
        double_tildes = buffer.count("~~")
        if double_tildes % 2 == 1:
            return True

        # check for incomplete links [text](url) or images ![text](url)
        open_brackets = buffer.count("[")
        complete_links = len(COMPLETE_LINKS_PATTERN.findall(buffer))
        complete_images = len(COMPLETE_IMAGES_PATTERN.findall(buffer))

        remaining_brackets = open_brackets - complete_links - complete_images
        if remaining_brackets > 0:
            return True

        return False

    def process_complete_text(text: str, is_newline: bool = False) -> str:
        if is_newline:
            for pattern, replacement in LINE_PATTERNS:
                text = pattern.sub(replacement, text)

        for pattern, replacement in INLINE_PATTERNS:
            text = pattern.sub(replacement, text)

        return text

    buffer = ""
    buffer_is_newline = True  # track if buffer is at start of line

    async for chunk in text:
        buffer += chunk

        if "\n" in buffer:
            lines = buffer.split("\n")
            buffer = lines[-1]  # keep last incomplete line

            for i, line in enumerate(lines[:-1]):
                is_newline = buffer_is_newline if i == 0 else True
                processed_line = process_complete_text(line, is_newline=is_newline)
                yield processed_line + "\n"

            buffer_is_newline = True
            continue

        # split at the position after the split token
        last_split_pos = 0
        for token in INLINE_SPLIT_TOKENS:
            last_split_pos = max(last_split_pos, buffer.rfind(token, last_split_pos))
            if last_split_pos >= len(buffer) - 1:
                break

        if last_split_pos >= 1:
            processable = buffer[:last_split_pos]  # exclude the split token
            rest = buffer[last_split_pos:]
            if not has_incomplete_pattern(processable):
                yield process_complete_text(processable, is_newline=buffer_is_newline)
                buffer = rest
                buffer_is_newline = False

    if buffer:
        yield process_complete_text(buffer, is_newline=buffer_is_newline)


# Unicode block ranges from: https://unicode.org/Public/UNIDATA/Blocks.txt
EMOJI_PATTERN = re.compile(
    r"[\U0001F000-\U0001FBFF]"  # Emoji blocks: Mahjong Tiles through Symbols for Legacy Computing
    r"|[\U00002600-\U000026FF]"  # Miscellaneous Symbols
    r"|[\U00002700-\U000027BF]"  # Dingbats
    r"|[\U00002B00-\U00002BFF]"  # Miscellaneous Symbols and Arrows
    r"|[\U0000FE00-\U0000FE0F]"  # Variation selectors
    r"|\U0000200D"  # Zero width joiner
    r"|\U000020E3"  # Combining enclosing keycap
    r"+",
    re.UNICODE,
)


async def filter_emoji(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Filter out emojis from the text.
    """

    async for chunk in text:
        filtered_chunk = EMOJI_PATTERN.sub("", chunk)
        yield filtered_chunk


async def format_numbers(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Format numbers for TTS:
    - Small numbers (0-99) converted to words
    - Large numbers (>=100) remain as digits
    - Decimals formatted as 'point'
    - Negative numbers get 'minus' prefix
    - Years (1900-2099) remain unchanged
    - Commas removed from numbers
    """

    def remove_commas(text_chunk: str) -> str:
        """Remove commas from number sequences."""
        for _ in range(4):  # Handle up to 4 comma groups
            text_chunk = re.sub(r"(\d),(\d{3})", r"\1\2", text_chunk)
        return text_chunk

    def process_number(match: re.Match[str]) -> str:
        """Process a single number match."""
        num_str = match.group(0)
        num = float(num_str)

        # Handle negative numbers
        if num_str.startswith("-"):
            positive = abs(num)
            if positive < 100:
                return "minus " + number_to_words(int(positive))
            return "minus " + str(int(positive))

        # Handle decimals
        if "." in num_str:
            parts = num_str.split(".")
            whole = parts[0] or "0"
            decimal = parts[1] or ""

            # Protect time-like formats (e.g., "14.00" for 2pm)
            if decimal == "00" and 0 <= int(whole) <= 23:
                return num_str

            return f"{whole} point {' '.join(list(decimal))}"

        # Preserve years
        if 1900 <= num <= 2099:
            return str(num_str)

        # Small numbers to words
        if 0 <= num < 100:
            return str(number_to_words(int(num)))

        return str(num_str)

    # Match negative numbers, decimals, and integers
    pattern = r"(?:^|(?<=\s))-?\d+(?:\.\d+)?(?=\s|[^\w:]|$)"

    async for chunk in _buffered_regex_filter(
        text, pattern, process_number, preprocess_chunk=remove_commas, flags=re.MULTILINE
    ):
        yield chunk


async def format_dollar_amounts(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Format dollar amounts for TTS:
    - $5 -> "five dollars"
    - $12.50 -> "twelve dollars and fifty cents"
    - $0.023 -> "zero point zero two three dollars" (speaks out each decimal digit)
    - Handles singular/plural correctly
    """

    def process_dollar(match: re.Match[str]) -> str:
        """Process a single dollar amount."""
        dollars = int(match.group(1))
        cents_str = match.group(2)

        # For small amounts with many decimal places, speak out each decimal digit
        if cents_str and len(cents_str) > 2:
            # Convert to "zero point zero two three dollars"
            dollar_words = number_to_words(dollars) if dollars < 100 else str(dollars)
            decimal_digits = " ".join(
                number_to_words(int(d)) if d != "0" else "zero" for d in cents_str
            )
            return f"{dollar_words} point {decimal_digits} dollars"

        result = number_to_words(dollars) if dollars < 100 else str(dollars)
        result += " dollar" if dollars == 1 else " dollars"

        if cents_str and int(cents_str) > 0:
            cent_num = int(cents_str)
            # Only format as cents if exactly 2 digits
            if len(cents_str) == 2:
                result += " and " + number_to_words(cent_num)
                result += " cent" if cent_num == 1 else " cents"

        return result

    async for chunk in _buffered_regex_filter(text, r"\$(\d+)(?:\.(\d+))?", process_dollar):
        yield chunk


async def format_percentages(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Format percentages for TTS:
    - 67% -> "67 percent"
    - 89.5% -> "89.5 percent" (keep decimal numeric to avoid mixed format)
    """
    async for chunk in _buffered_regex_filter(
        text, r"\b(\d+(?:\.\d+)?)%", lambda m: f"{m.group(1)} percent"
    ):
        yield chunk


async def format_distances(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Format distances for TTS:
    - 5 km -> "5 kilometers"
    - 15.3 mi -> "15.3 miles" (keep decimal numeric)
    - 1,500 m -> "1500 meters" (remove commas, keep numeric)
    """
    units = {
        "km": "kilometers",
        "mi": "miles",
        "m": "meters",
        "ft": "feet",
        "yd": "yards",
    }

    def process_distance(match: re.Match[str]) -> str:
        """Process a single distance measurement."""
        num = match.group(1).replace(",", "")  # Remove commas
        unit = match.group(2).lower()
        full_unit = units.get(unit, unit)
        return f"{num} {full_unit}"

    async for chunk in _buffered_regex_filter(
        text,
        r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(km|mi|m|ft|yd)\b",
        process_distance,
        flags=re.IGNORECASE,
    ):
        yield chunk


async def format_units(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Format weight/volume units for TTS:
    - 10 kg -> "ten kilograms"
    - 85.6 kg -> "85.6 kilograms" (keep decimal numeric)
    - 1,500 kg -> "1500 kilograms" (remove commas, keep numeric)
    """
    units = {
        "lb": "pounds",
        "lbs": "pounds",
        "oz": "ounces",
        "kg": "kilograms",
        "g": "grams",
        "mg": "milligrams",
        "l": "liters",
        "ml": "milliliters",
        "gal": "gallons",
    }

    def process_unit(match: re.Match[str]) -> str:
        """Process a single unit measurement."""
        num_str = match.group(1).replace(",", "")  # Remove commas
        unit = match.group(2).lower()
        full_unit = units.get(unit, unit)

        # Keep decimals numeric; for small integers, convert to words
        if "." in num_str:
            return f"{num_str} {full_unit}"

        num = int(num_str)
        if num < 100:
            return f"{number_to_words(num)} {full_unit}"
        return f"{num_str} {full_unit}"

    async for chunk in _buffered_regex_filter(
        text,
        r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(lb|lbs|oz|kg|g|mg|l|ml|gal)\b",
        process_unit,
        flags=re.IGNORECASE,
    ):
        yield chunk


async def format_emails(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Format email addresses for TTS:
    - john.doe@example.com -> "john dot doe at example dot com"
    """

    def process_email(match: re.Match[str]) -> str:
        local = match.group(1).replace(".", " dot ")
        domain = match.group(2).replace(".", " dot ")
        return f"{local} at {domain}"

    async for chunk in _buffered_regex_filter(
        text, r"([a-zA-Z0-9._+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", process_email
    ):
        yield chunk


async def format_phone_numbers(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Format phone numbers for TTS:
    - 555-123-4567 -> "5 5 5 1 2 3 4 5 6 7"
    """

    def process_phone(match: re.Match[str]) -> str:
        digits = match.group(1) + match.group(2) + match.group(3)
        return " ".join(list(digits))

    async for chunk in _buffered_regex_filter(
        text, r"\b(\d{3})[-.]?(\d{3})[-.]?(\d{4})\b", process_phone
    ):
        yield chunk


async def format_dates(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Format dates for TTS:
    - 2024-12-25 -> "Wednesday, December 25, 2024"
    """
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def process_date(match: re.Match[str]) -> str:
        from datetime import datetime

        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))

        try:
            date = datetime(year, month, day)
            day_name = days[date.weekday()]
            month_name = months[date.month - 1]
            return f"{day_name}, {month_name} {day}, {year}"
        except (ValueError, IndexError):
            return str(match.group(0))

    async for chunk in _buffered_regex_filter(
        text, r"\b(\d{4})[\s-](\d{2})[\s-](\d{2})\b", process_date
    ):
        yield chunk


async def format_times(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Format times for TTS.

    This filter simplifies time formats when minutes/seconds are 00.

    Examples:
    - "14:00" -> "14" (simplified when minutes are 00)
    - "14:30" -> "14:30" (kept as-is)
    - "Meeting at 2:00" -> "Meeting at 2"
    """

    def process_time(match: re.Match[str]) -> str:
        hours = match.group(1)
        minutes = match.group(2)
        seconds = match.group(3)

        # If minutes and seconds are 00, just return hours
        if minutes == "00" and (not seconds or seconds == "00"):
            return str(hours)

        # Otherwise keep as-is
        return str(match.group(0))

    async for chunk in _buffered_regex_filter(
        text, r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", process_time
    ):
        yield chunk


async def format_acronyms(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Format acronyms for TTS:
    - NASA, API -> lowercase (nasa, api)
    - XYZ (no vowels) -> space out (X Y Z)
    """

    def process_acronym(match: re.Match[str]) -> str:
        stripped = match.group(0).replace(".", "")
        if len(stripped) < 2:
            return str(match.group(0))

        # Known acronyms -> lowercase
        if stripped in KNOWN_ACRONYMS:
            return str(match.group(0).lower())

        # Has vowels -> lowercase
        if re.search(r"[AEIOUY]", stripped, re.IGNORECASE):
            return str(match.group(0).lower())

        # No vowels -> space out
        return str(" ".join(list(stripped)))

    async for chunk in _buffered_regex_filter(text, r"\b[A-Z]+(?:\.[A-Z]+)*\b", process_acronym):
        yield chunk


async def remove_angle_bracket_content(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """Remove HTML-like tags except special TTS tags like <break>."""
    async for chunk in _buffered_regex_filter(
        text, r"<(?!break|spell|<)(?![^>]*>>)[^>]*>", lambda m: "", flags=re.IGNORECASE
    ):
        yield chunk


async def replace_newlines_with_periods(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Replace newlines with periods or spaces for smoother TTS flow.

    This filter helps create natural speech pauses by converting line breaks
    into punctuation that TTS engines can interpret.

    Examples:
    - "Hello\n\nWorld" -> "Hello. World" (multiple newlines become period+space)
    - "Hello\nWorld" -> "Hello World" (single newline becomes space)

    This is particularly useful for processing LLM outputs that may include
    markdown-style formatting with line breaks.
    """
    async for chunk in text:
        # Multiple newlines -> period + space
        result = re.sub(r"\n\s*\n+", ". ", chunk)
        # Single newline -> space
        result = result.replace("\n", " ")
        yield result
