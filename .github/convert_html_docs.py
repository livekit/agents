#!/usr/bin/env python3
"""Convert pdoc3 HTML API reference docs to markdown.

This script converts the already-built pdoc3 HTML docs (from CloudFront/S3) into
clean markdown files suitable for consumption by LLMs via the MCP server.

The generated files match the ``python-md/`` convention expected by
``python-reference.ts`` in the docs MCP server.

Path mapping (canonical, no ambiguity):
    livekit/index.html        -> livekit.md
    livekit/agents/index.html -> livekit/agents.md
    livekit/agents/vad.html   -> livekit/agents/vad.md

Usage:
    uv run python .github/convert_html_docs.py INPUT_DIR OUTPUT_DIR [options]
      --validate-only    Only validate existing markdown against HTML
      --no-validate      Skip validation after conversion
      --verbose / -v     Show per-file validation details
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from markdownify import MarkdownConverter

# ---------------------------------------------------------------------------
# Path mapping
# ---------------------------------------------------------------------------


def map_output_path(html_path: Path, input_dir: Path) -> Path:
    """Map an HTML input path to its markdown output path.

    Canonical rule (zero collisions across all 264 Python HTML files):
        dir/index.html  -> dir.md
        dir/file.html   -> dir/file.md

    Raises ValueError for paths containing ``..`` or other unsafe components.
    """
    rel = html_path.relative_to(input_dir)

    # Path safety: reject traversal
    for part in rel.parts:
        if part in ("..", "~") or part.startswith(".."):
            raise ValueError(f"Unsafe path component in {rel}: {part!r}")

    parts = list(rel.parts)

    # Strip leading "livekit" prefix directory if present — the HTML bucket
    # has  livekit/agents/index.html  but we want  livekit/agents.md
    # (we do NOT strip "livekit" — keep it in output path)

    if parts[-1] == "index.html":
        # dir/index.html -> dir.md
        parts = parts[:-1]
        if not parts:
            return Path("index.md")
        return (
            Path(*parts[:-1], parts[-1] + ".md")
            if len(parts) == 1
            else Path(*parts[:-1], parts[-1] + ".md")
        )
    else:
        # dir/file.html -> dir/file.md
        stem = parts[-1].removesuffix(".html")
        parts[-1] = stem + ".md"
        return Path(*parts)


def map_link(href: str) -> str:
    """Convert a relative .html link to .md for cross-references."""
    if not href or href.startswith(("http://", "https://", "#", "mailto:")):
        return href
    # Strip fragment
    base, _, fragment = href.partition("#")
    if not base.endswith(".html"):
        return href
    # index.html -> parent .md
    if base.endswith("/index.html"):
        base = base[: -len("/index.html")] + ".md"
    elif base == "index.html":
        base = "../index.md"
    else:
        base = base.removesuffix(".html") + ".md"
    return base + (f"#{fragment}" if fragment else "")


# ---------------------------------------------------------------------------
# PdocMarkdownConverter — markdownify subclass for <div class="desc"> content
# ---------------------------------------------------------------------------

_ORPHAN_REF_LINK_RE = re.compile(r"\[([^\]]*)\]\[([^\]]*)\]")


class PdocMarkdownConverter(MarkdownConverter):
    """Custom markdownify converter for pdoc3 description blocks."""

    def __init__(self, *, page_converter: PageConverter | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._page_converter = page_converter

    def convert_details(self, el: Tag, text: str, parent_tags: set[str] | None = None) -> str:  # type: ignore[override]
        """Skip <details class="source"> (handled separately by PageConverter)."""
        if "source" in (el.get("class") or []):
            return ""
        return text

    def convert_div(self, el: Tag, text: str, parent_tags: set[str] | None = None) -> str:  # type: ignore[override]
        classes = el.get("class") or []
        if "admonition" in classes:
            title_el = el.find(class_="admonition-title")
            title = title_el.get_text(strip=True) if title_el else "Note"
            # Remove the title element text from the body
            body = text
            if title_el:
                body = body.replace(title_el.get_text(), "", 1).strip()
            lines = [
                f"> **{title}:** {line}" if i == 0 else f"> {line}"
                for i, line in enumerate(body.split("\n"))
                if line.strip()
            ]
            return "\n".join(lines) + "\n\n"
        return text

    def convert_pre(self, el: Tag, text: str, parent_tags: set[str] | None = None) -> str:  # type: ignore[override]
        code_el = el.find("code")
        if code_el and isinstance(code_el, Tag):
            classes = code_el.get("class") or []
            lang = ""
            for cls in classes:
                if isinstance(cls, str) and cls != "hljs":
                    lang = cls.removeprefix("language-")
                    break
            code_text = code_el.get_text()
            return f"\n```{lang}\n{code_text}\n```\n"
        return f"\n```\n{text}\n```\n"

    def convert_a(self, el: Tag, text: str, parent_tags: set[str] | None = None) -> str:  # type: ignore[override]
        href = el.get("href", "")
        if isinstance(href, list):
            href = href[0] if href else ""
        if self._page_converter is not None:
            href = self._page_converter._rewrite_link(href)
        else:
            href = map_link(href)
        title = el.get("title", "")
        if not href:
            return text
        if title:
            return f'[{text}]({href} "{title}")'
        return f"[{text}]({href})"

    def convert_dl(self, el: Tag, text: str, parent_tags: set[str] | None = None) -> str:  # type: ignore[override]
        """Convert definition lists (e.g. parameter docs) to markdown lists."""
        lines: list[str] = []
        for child in el.children:
            if isinstance(child, Tag):
                if child.name == "dt":
                    code = child.find("code")
                    if code:
                        term = code.get_text(strip=True)
                        lines.append(f"- **`{term}`**")
                    else:
                        term = child.get_text(strip=True)
                        lines.append(f"- **{term}**")
                elif child.name == "dd":
                    # Use markdownify to preserve spaces around inline <code>
                    # elements and properly convert them to backticks.
                    inner_html = child.decode_contents()
                    desc = MarkdownConverter().convert(inner_html).strip()
                    # Collapse to single line for list-item format
                    desc = " ".join(desc.split())
                    # Escape orphan reference-link patterns ([text][ref])
                    # from unresolved docstring cross-references (e.g. Pydantic)
                    desc = _ORPHAN_REF_LINK_RE.sub(r"\\[\1\\]\\[\2\\]", desc)
                    if desc and lines:
                        lines[-1] += f": {desc}"
        return "\n".join(lines) + "\n\n" if lines else text


def _desc_to_markdown(el: Tag | None, *, page_converter: PageConverter | None = None) -> str:
    """Convert a <div class="desc"> element to markdown."""
    if el is None:
        return ""
    text = PdocMarkdownConverter(
        page_converter=page_converter,
        heading_style="ATX",
        bullets="-",
        strip=["img"],
    ).convert(str(el))
    return text.strip()


# ---------------------------------------------------------------------------
# Signature / source extraction helpers
# ---------------------------------------------------------------------------

_NBSP_RE = re.compile(r"\u00a0|\u2011|\u2010|\u2012|\u2013")


def _normalize_sig(text: str) -> str:
    """Normalize a signature string: unescape HTML, fix non-breaking hyphens."""
    text = html.unescape(text)
    text = _NBSP_RE.sub(lambda m: "-" if m.group() in "\u2011\u2010\u2012\u2013" else " ", text)
    text = text.replace("\u2002", " ").replace("\u2003", " ")  # en/em space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_signature(dt: Tag) -> str:
    """Extract a function/class signature from a <dt> element.

    Handles <br> tags (multi-line signatures) and non-breaking hyphens.
    """
    code = dt.find("code", class_="name")
    if not code:
        return ""
    # Replace <br> with newline before extracting text
    for br in code.find_all("br"):
        br.replace_with("\n")
    raw = code.get_text()
    return _normalize_sig(raw)


def _extract_source(dd: Tag) -> str | None:
    """Extract source code from <details class="source"> inside a <dd>."""
    details = dd.find("details", class_="source")
    if not details or not isinstance(details, Tag):
        return None
    pre = details.find("pre")
    if not pre or not isinstance(pre, Tag):
        return None
    code_el = pre.find("code")
    if not code_el:
        return None
    raw = code_el.get_text() if isinstance(code_el, Tag) else str(code_el)
    return html.unescape(raw).strip() or None


def _extract_ident(dt: Tag) -> str:
    """Extract the short identifier name from a <dt> element."""
    span = dt.find("span", class_="ident")
    if span:
        return span.get_text(strip=True)
    return ""


# ---------------------------------------------------------------------------
# PageConverter — structured DOM walker
# ---------------------------------------------------------------------------


class PageConverter:
    """Convert a single pdoc3 HTML page to markdown."""

    def __init__(
        self,
        html_content: str,
        *,
        html_rel_path: Path | None = None,
        input_dir: Path | None = None,
    ) -> None:
        self.soup = BeautifulSoup(html_content, "html.parser")
        self.lines: list[str] = []
        # Context for link rewriting — when set, links are resolved relative to
        # the source HTML file's location and re-computed relative to the output
        # markdown file's location.
        self._html_rel_path = html_rel_path
        self._input_dir = input_dir

    def _rewrite_link(self, href: str) -> str:
        """Rewrite an HTML href to a correct relative .md path.

        When source context is available, resolves the link against the source
        HTML file's directory, maps through ``map_output_path``, and computes
        the correct relative path from the output markdown file.

        Falls back to the context-free ``map_link`` when source context is
        unavailable.
        """
        if not href or href.startswith(("http://", "https://", "#", "mailto:")):
            return href
        base, _, fragment = href.partition("#")
        if not base.endswith(".html"):
            return href

        if self._html_rel_path is None or self._input_dir is None:
            return map_link(href)

        # Resolve the target HTML path relative to the source HTML's directory
        source_html_dir = self._html_rel_path.parent
        target_html_abs = (self._input_dir / source_html_dir / base).resolve()
        # Make it relative to input_dir again
        try:
            target_html_rel = target_html_abs.relative_to(self._input_dir.resolve())
        except ValueError:
            return map_link(href)

        # Map both source and target through map_output_path
        try:
            source_md = map_output_path(self._input_dir / self._html_rel_path, self._input_dir)
            target_md = map_output_path(self._input_dir / target_html_rel, self._input_dir)
        except ValueError:
            return map_link(href)

        # Compute the relative path from source .md's directory to target .md
        rel = os.path.relpath(str(target_md), str(source_md.parent))
        result = rel
        if fragment:
            result += f"#{fragment}"
        return result

    def convert(self) -> str:
        article = self.soup.find("article", id="content")
        if not article or not isinstance(article, Tag):
            return ""

        self._convert_title(article)
        self._convert_intro(article)
        self._convert_submodules(article)
        self._convert_variables(article)
        self._convert_functions(article)
        self._convert_classes(article)

        return "\n".join(self.lines).rstrip() + "\n"

    # --- Title ---

    def _convert_title(self, article: Tag) -> None:
        h1 = article.find("h1", class_="title")
        if not h1 or not isinstance(h1, Tag):
            return
        code = h1.find("code")
        if code:
            module_name = code.get_text(strip=True)
        else:
            module_name = h1.get_text(strip=True)
        # Determine type prefix (Module, Package, Namespace)
        full_text = h1.get_text(strip=True)
        prefix = "Module"
        for p in ("Namespace", "Package", "Module"):
            if full_text.startswith(p):
                prefix = p
                break
        self.lines.append(f"# {prefix} `{module_name}`")
        self.lines.append("")

    # --- Section intro ---

    def _convert_intro(self, article: Tag) -> None:
        intro = article.find("section", id="section-intro")
        if not intro or not isinstance(intro, Tag):
            return
        # Convert the intro section content (skip the section tag itself)
        text = _desc_to_markdown(intro, page_converter=self)
        if text:
            self.lines.append(text)
            self.lines.append("")

    # --- Sub-modules ---

    def _convert_submodules(self, article: Tag) -> None:
        h2 = article.find("h2", id="header-submodules")
        if not h2 or not isinstance(h2, Tag):
            return
        self.lines.append("## Sub-modules")
        self.lines.append("")

        # The <dl> follows the <h2> in the next <section> or same section
        section = h2.find_parent("section")
        if not section:
            return
        dl = section.find("dl")
        if not dl or not isinstance(dl, Tag):
            return

        for dt in dl.find_all("dt", recursive=False):
            a = dt.find("a")
            if not a:
                continue
            name = a.get_text(strip=True)
            href = self._rewrite_link(a.get("href", ""))
            dd = dt.find_next_sibling("dd")
            desc_el = dd.find("div", class_="desc") if dd else None
            desc_text = ""
            if desc_el and isinstance(desc_el, Tag):
                desc_text = " ".join(desc_el.get_text().strip().split())

            if desc_text:
                self.lines.append(f"- [`{name}`]({href}) - {desc_text}")
            else:
                self.lines.append(f"- [`{name}`]({href})")

        self.lines.append("")

    # --- Variables ---

    def _convert_variables(self, article: Tag) -> None:
        h2 = article.find("h2", id="header-variables")
        if not h2 or not isinstance(h2, Tag):
            return
        self.lines.append("## Global variables")
        self.lines.append("")

        section = h2.find_parent("section")
        if not section:
            return
        dl = section.find("dl")
        if not dl or not isinstance(dl, Tag):
            return

        for dt in dl.find_all("dt", recursive=False):
            self._convert_variable_or_property(dt, heading_level=3)

    # --- Functions ---

    def _convert_functions(self, article: Tag) -> None:
        h2 = article.find("h2", id="header-functions")
        if not h2 or not isinstance(h2, Tag):
            return
        self.lines.append("## Functions")
        self.lines.append("")

        section = h2.find_parent("section")
        if not section:
            return
        dl = section.find("dl")
        if not dl or not isinstance(dl, Tag):
            return

        for dt in dl.find_all("dt", recursive=False):
            self._convert_function(dt, heading_level=3)

    def _convert_function(self, dt: Tag, heading_level: int) -> None:
        name = _extract_ident(dt)
        if not name:
            return
        sig = _extract_signature(dt)
        dd = dt.find_next_sibling("dd")
        full_id = dt.get("id", "")

        self.lines.append(f"{'#' * heading_level} `{name}`")
        self.lines.append("")
        if full_id:
            self.lines.append(f"Full name: `{full_id}`")
            self.lines.append("")
        if sig:
            self.lines.append(f"```python\n{sig}\n```")
            self.lines.append("")

        if dd and isinstance(dd, Tag):
            # Description
            desc_el = dd.find("div", class_="desc", recursive=False)
            if desc_el and isinstance(desc_el, Tag):
                desc_text = _desc_to_markdown(desc_el, page_converter=self)
                if desc_text:
                    self.lines.append(desc_text)
                    self.lines.append("")

            # Source
            source = _extract_source(dd)
            if source:
                self.lines.append("<details><summary>Source</summary>")
                self.lines.append("")
                self.lines.append(f"```python\n{source}\n```")
                self.lines.append("")
                self.lines.append("</details>")
                self.lines.append("")

    # --- Classes ---

    def _convert_classes(self, article: Tag) -> None:
        h2 = article.find("h2", id="header-classes")
        if not h2 or not isinstance(h2, Tag):
            return
        self.lines.append("## Classes")
        self.lines.append("")

        section = h2.find_parent("section")
        if not section:
            return
        dl = section.find("dl")
        if not dl or not isinstance(dl, Tag):
            return

        for dt in dl.find_all("dt", recursive=False):
            self._convert_class(dt)

    def _convert_class(self, dt: Tag) -> None:
        name = _extract_ident(dt)
        if not name:
            return
        sig = _extract_signature(dt)
        dd = dt.find_next_sibling("dd")
        full_id = dt.get("id", "")

        self.lines.append(f"### `{name}`")
        self.lines.append("")
        if full_id:
            self.lines.append(f"Full name: `{full_id}`")
            self.lines.append("")
        if sig:
            self.lines.append(f"```python\n{sig}\n```")
            self.lines.append("")

        if not dd or not isinstance(dd, Tag):
            return

        # Class description
        desc_el = dd.find("div", class_="desc", recursive=False)
        if desc_el and isinstance(desc_el, Tag):
            desc_text = _desc_to_markdown(desc_el, page_converter=self)
            if desc_text:
                self.lines.append(desc_text)
                self.lines.append("")

        # Source
        source = _extract_source(dd)
        if source:
            self.lines.append("<details><summary>Source</summary>")
            self.lines.append("")
            self.lines.append(f"```python\n{source}\n```")
            self.lines.append("")
            self.lines.append("</details>")
            self.lines.append("")

        # Walk h3 subsections within this <dd>
        for h3 in dd.find_all("h3", recursive=False):
            section_title = h3.get_text(strip=True)
            self._convert_class_section(dd, h3, section_title)

    def _convert_class_section(self, dd: Tag, h3: Tag, title: str) -> None:
        title_lower = title.lower()

        if title_lower in ("ancestors", "subclasses"):
            self.lines.append(f"#### {title}")
            self.lines.append("")
            ul = h3.find_next_sibling("ul")
            if ul and isinstance(ul, Tag):
                for li in ul.find_all("li"):
                    text = li.get_text(strip=True)
                    self.lines.append(f"- {text}")
                self.lines.append("")

        elif title_lower in (
            "class variables",
            "instance variables",
            "static methods",
        ):
            self.lines.append(f"#### {title}")
            self.lines.append("")
            dl = h3.find_next_sibling("dl")
            if dl and isinstance(dl, Tag):
                for sub_dt in dl.find_all("dt", recursive=False):
                    self._convert_variable_or_property(sub_dt, heading_level=5)

        elif title_lower == "methods":
            self.lines.append("#### Methods")
            self.lines.append("")
            dl = h3.find_next_sibling("dl")
            if dl and isinstance(dl, Tag):
                for sub_dt in dl.find_all("dt", recursive=False):
                    self._convert_function(sub_dt, heading_level=5)

        elif title_lower == "inherited members":
            self.lines.append("#### Inherited members")
            self.lines.append("")
            ul = h3.find_next_sibling("ul")
            if ul and isinstance(ul, Tag):
                for li in ul.find_all("li", recursive=False):
                    # Each top-level li has a parent class link + nested ul of members
                    parent_code = li.find("code", recursive=False)
                    if not parent_code:
                        parent_code = li.find("b")
                    parent_link = parent_code.find("a") if parent_code else None
                    parent_name = (
                        parent_link.get_text(strip=True)
                        if parent_link
                        else (parent_code.get_text(strip=True) if parent_code else "")
                    )
                    nested_ul = li.find("ul")
                    members: list[str] = []
                    if nested_ul and isinstance(nested_ul, Tag):
                        for member_li in nested_ul.find_all("li"):
                            members.append(member_li.get_text(strip=True))
                    if members:
                        self.lines.append(f"- **{parent_name}**: {', '.join(members)}")
                    else:
                        self.lines.append(f"- **{parent_name}**")
                self.lines.append("")

        else:
            # Fallback: just emit the heading
            self.lines.append(f"#### {title}")
            self.lines.append("")

    def _convert_variable_or_property(self, dt: Tag, heading_level: int) -> None:
        """Convert a class/instance variable or property <dt>/<dd> pair."""
        name = _extract_ident(dt)
        if not name:
            return
        full_id = dt.get("id", "")
        # Get the full signature line (var/prop name : type)
        code = dt.find("code", class_="name")
        sig_text = ""
        if code:
            sig_text = _normalize_sig(code.get_text())

        dd = dt.find_next_sibling("dd")

        self.lines.append(f"{'#' * heading_level} `{name}`")
        self.lines.append("")
        if full_id:
            self.lines.append(f"Full name: `{full_id}`")
            self.lines.append("")
        if sig_text:
            self.lines.append(f"`{sig_text}`")
            self.lines.append("")

        if dd and isinstance(dd, Tag):
            desc_el = dd.find("div", class_="desc", recursive=False)
            if desc_el and isinstance(desc_el, Tag):
                desc_text = _desc_to_markdown(desc_el, page_converter=self)
                if desc_text:
                    self.lines.append(desc_text)
                    self.lines.append("")

            source = _extract_source(dd)
            if source:
                self.lines.append("<details><summary>Source</summary>")
                self.lines.append("")
                self.lines.append(f"```python\n{source}\n```")
                self.lines.append("")
                self.lines.append("</details>")
                self.lines.append("")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ContentFingerprint:
    """Structural fingerprint extracted from HTML for validation."""

    identifiers: list[str] = field(default_factory=list)
    source_blocks: list[str] = field(default_factory=list)
    signatures: list[str] = field(default_factory=list)
    description_texts: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of validating markdown against an HTML fingerprint."""

    is_valid: bool
    score: float
    identifier_found: int = 0
    identifier_total: int = 0
    source_found: int = 0
    source_total: int = 0
    signature_found: int = 0
    signature_total: int = 0
    description_score: float = 1.0
    missing_identifiers: list[str] = field(default_factory=list)
    missing_sources: int = 0
    missing_signatures: list[str] = field(default_factory=list)
    description_warnings: list[str] = field(default_factory=list)


def extract_fingerprint(html_content: str) -> ContentFingerprint:
    """Extract a structural fingerprint from pdoc3 HTML for validation."""
    soup = BeautifulSoup(html_content, "html.parser")
    fp = ContentFingerprint()

    # Identifiers: all dt[id] values
    for dt in soup.find_all("dt", id=True):
        fp.identifiers.append(dt["id"])

    # Source blocks
    for details in soup.find_all("details", class_="source"):
        pre = details.find("pre")
        if pre:
            code_el = pre.find("code")
            if code_el:
                raw = code_el.get_text() if isinstance(code_el, Tag) else str(code_el)
                fp.source_blocks.append(html.unescape(raw).strip())

    # Signatures: function/class names from <dt><code class="name"><span class="ident">
    for dt in soup.find_all("dt"):
        span = dt.find("span", class_="ident")
        if span:
            fp.signatures.append(span.get_text(strip=True))

    # Description texts
    for desc in soup.find_all("div", class_="desc"):
        text = desc.get_text(strip=True)
        if text:
            fp.description_texts.append(text)

    return fp


def _normalize_ws(text: str) -> str:
    """Normalize whitespace for comparison."""
    return re.sub(r"\s+", " ", text).strip()


def validate_markdown(fingerprint: ContentFingerprint, markdown: str) -> ValidationResult:
    """Validate that markdown contains the expected content from the HTML.

    Exact checks (identifiers, sources, signatures) are gates.
    Fuzzy description overlap is warning-only.
    """
    result = ValidationResult(is_valid=True, score=1.0)

    # 1. Identifier check — exact substring match (gates)
    result.identifier_total = len(fingerprint.identifiers)
    for ident in fingerprint.identifiers:
        if ident in markdown:
            result.identifier_found += 1
        else:
            result.missing_identifiers.append(ident)

    # 2. Source block check — whitespace-normalized line-by-line (gates)
    result.source_total = len(fingerprint.source_blocks)
    for source in fingerprint.source_blocks:
        norm_source_lines = [_normalize_ws(line) for line in source.split("\n") if line.strip()]
        # Check that all significant lines appear in the markdown
        found = True
        for line in norm_source_lines[:5]:  # check first 5 lines for efficiency
            if len(line) > 10 and _normalize_ws(line) not in _normalize_ws(markdown):
                found = False
                break
        if found:
            result.source_found += 1
        else:
            result.missing_sources += 1

    # 3. Signature check — function/class name must appear (gates)
    result.signature_total = len(fingerprint.signatures)
    for sig_name in fingerprint.signatures:
        if sig_name in markdown:
            result.signature_found += 1
        else:
            result.missing_signatures.append(sig_name)

    # 4. Description check — fuzzy token overlap (warning-only, does NOT gate)
    if fingerprint.description_texts:
        desc_scores: list[float] = []
        for desc_text in fingerprint.description_texts:
            tokens = {w for w in re.findall(r"\w+", desc_text.lower()) if len(w) >= 4}
            if not tokens:
                desc_scores.append(1.0)
                continue
            found_tokens = sum(1 for t in tokens if t in markdown.lower())
            overlap = found_tokens / len(tokens)
            if overlap < 0.9:
                result.description_warnings.append(
                    f"Low overlap ({overlap:.0%}): {desc_text[:80]}..."
                )
            desc_scores.append(overlap)
        result.description_score = sum(desc_scores) / len(desc_scores)

    # Compute gate result: exact checks only
    id_ok = result.identifier_found == result.identifier_total
    source_ok = result.source_found == result.source_total
    sig_ok = result.signature_found == result.signature_total
    result.is_valid = id_ok and source_ok and sig_ok

    # Compute overall score (for reporting)
    counts = [
        (result.identifier_found, result.identifier_total),
        (result.source_found, result.source_total),
        (result.signature_found, result.signature_total),
    ]
    total = sum(t for _, t in counts)
    found = sum(f for f, _ in counts)
    result.score = found / total if total > 0 else 1.0

    return result


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


def convert_file(html_path: Path) -> str:
    """Convert a single HTML file to markdown."""
    html_content = html_path.read_text(encoding="utf-8")
    converter = PageConverter(html_content)
    return converter.convert()


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    validate: bool = True,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """Convert all HTML files in input_dir to markdown in output_dir.

    Returns (total_files, valid_files, failed_files).
    """
    html_files = sorted(input_dir.rglob("*.html"))
    if not html_files:
        print(f"No HTML files found in {input_dir}")
        return 0, 0, 0

    total = 0
    valid = 0
    failed = 0

    for html_path in html_files:
        try:
            out_path = map_output_path(html_path, input_dir)
        except ValueError as e:
            print(f"  SKIP {html_path}: {e}")
            continue

        total += 1
        full_out = output_dir / out_path
        full_out.parent.mkdir(parents=True, exist_ok=True)

        html_content = html_path.read_text(encoding="utf-8")
        html_rel = html_path.relative_to(input_dir)
        markdown = PageConverter(
            html_content,
            html_rel_path=html_rel,
            input_dir=input_dir,
        ).convert()
        full_out.write_text(markdown, encoding="utf-8")

        if validate:
            fp = extract_fingerprint(html_content)
            vr = validate_markdown(fp, markdown)
            if vr.is_valid:
                valid += 1
                if verbose:
                    status = "VALID"
                    if vr.description_warnings:
                        status += f" (desc warnings: {len(vr.description_warnings)})"
                    print(f"  {status} {out_path} (score={vr.score:.2f})")
            else:
                failed += 1
                print(f"  FAIL  {out_path} (score={vr.score:.2f})")
                if verbose:
                    if vr.missing_identifiers:
                        print(f"        missing identifiers: {vr.missing_identifiers[:5]}")
                    if vr.missing_sources:
                        print(f"        missing sources: {vr.missing_sources}")
                    if vr.missing_signatures:
                        print(f"        missing signatures: {vr.missing_signatures[:5]}")
        else:
            valid += 1
            if verbose:
                print(f"  OK    {out_path}")

    return total, valid, failed


# ---------------------------------------------------------------------------
# Index generation
# ---------------------------------------------------------------------------


def generate_index(input_dir: Path, output_dir: Path) -> None:
    """Generate an index.md listing all converted modules."""
    md_files = sorted(output_dir.rglob("*.md"))
    lines = [
        "# LiveKit Python SDK API Reference",
        "",
        "Auto-generated API reference for the LiveKit Python SDK and plugins.",
        "",
        "## Modules",
        "",
    ]
    for md_file in md_files:
        rel = md_file.relative_to(output_dir)
        if str(rel) == "index.md":
            continue
        # Convert path back to module name for display
        module_name = str(rel).removesuffix(".md").replace("/", ".")
        lines.append(f"- [`{module_name}`]({rel})")

    lines.append("")
    index_path = output_dir / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Link integrity validation
# ---------------------------------------------------------------------------

_MD_LINK_RE = re.compile(r"\[.*?\]\(([^)]+)\)")

# Known external hrefs from third-party docstrings.  Matched against the raw
# href (before resolution) stripped of any fragment.  Be specific — each entry
# is a literal string, not a pattern, so we never accidentally suppress a real
# broken link.
#
# To find new entries: run with --check-links -v after adding a new dependency
# that injects cross-site links in its docstrings.
_KNOWN_EXTERNAL_HREFS: set[str] = {
    # Pydantic BaseModel docstring links to docs.pydantic.dev/concepts/models
    "../concepts/models.md",
}


@dataclass
class LinkCheckResult:
    """Result of validating internal links across generated markdown files."""

    broken: list[tuple[Path, str]] = field(default_factory=list)
    """Links whose target file should exist but doesn't."""

    external: list[tuple[Path, str]] = field(default_factory=list)
    """Links whose raw href matches a known third-party docstring cross-reference
    (e.g. Pydantic's ``../concepts/models.md``).  Not failures."""


def validate_links(output_dir: Path) -> LinkCheckResult:
    """Walk all .md files and check that internal .md links resolve to existing files.

    Returns a ``LinkCheckResult`` with broken and external links separated.
    """
    result = LinkCheckResult()
    resolved_root = output_dir.resolve()

    for md_file in sorted(output_dir.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        for match in _MD_LINK_RE.finditer(content):
            href = match.group(1)
            # Strip title from link: [text](url "title")
            href = href.split('"')[0].strip()
            # Skip external URLs, anchors, non-.md links
            if href.startswith(("http://", "https://", "#", "mailto:")):
                continue
            # Split off fragment for file-existence check
            base = href.split("#")[0]
            if not base:
                continue
            if not base.endswith(".md"):
                continue

            # Resolve relative to the source file's directory
            target = (md_file.parent / base).resolve()
            rel_source = md_file.relative_to(output_dir)

            # Links that escape the output tree are always broken.
            try:
                target.relative_to(resolved_root)
            except ValueError:
                result.broken.append((rel_source, href))
                continue

            # If the target file exists, the link is valid regardless of
            # whether the href matches a known-external pattern.
            if target.exists():
                continue

            # Target is missing.  If the href matches a known third-party
            # docstring cross-reference, classify as external; otherwise
            # it's a broken internal link.
            if base in _KNOWN_EXTERNAL_HREFS:
                result.external.append((rel_source, href))
            else:
                result.broken.append((rel_source, href))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert pdoc3 HTML docs to markdown for LLM consumption."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing HTML docs")
    parser.add_argument("output_dir", type=Path, help="Directory to write markdown files")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing markdown against HTML (no conversion)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after conversion",
    )
    parser.add_argument(
        "--check-links",
        action="store_true",
        help="Validate that all internal .md links resolve to existing files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-file validation details",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        return 1

    if args.validate_only:
        if not output_dir.is_dir():
            print(f"Error: {output_dir} is not a directory", file=sys.stderr)
            return 1
        # Validate existing markdown against HTML
        html_files = sorted(input_dir.rglob("*.html"))
        total = 0
        valid = 0
        failed = 0
        for html_path in html_files:
            try:
                out_path = map_output_path(html_path, input_dir)
            except ValueError:
                continue
            md_path = output_dir / out_path
            if not md_path.exists():
                print(f"  MISSING {out_path}")
                failed += 1
                total += 1
                continue
            total += 1
            html_content = html_path.read_text(encoding="utf-8")
            markdown = md_path.read_text(encoding="utf-8")
            fp = extract_fingerprint(html_content)
            vr = validate_markdown(fp, markdown)
            if vr.is_valid:
                valid += 1
                if args.verbose:
                    print(f"  VALID {out_path} (score={vr.score:.2f})")
            else:
                failed += 1
                print(f"  FAIL  {out_path} (score={vr.score:.2f})")
                if args.verbose and vr.missing_identifiers:
                    print(f"        missing: {vr.missing_identifiers[:5]}")
        print(f"\nValidation: {valid}/{total} valid, {failed} failed")
        return 0 if failed == 0 else 1

    # Convert
    print(f"Converting {input_dir} -> {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    total, valid, failed = convert_directory(
        input_dir,
        output_dir,
        validate=not args.no_validate,
        verbose=args.verbose,
    )

    # Generate index
    generate_index(input_dir, output_dir)
    total += 1  # count index
    valid += 1

    print(f"\nDone! Converted {total} files to {output_dir}/")
    if not args.no_validate:
        print(f"Validation: {valid}/{total} valid, {failed} failed")
        pct = valid / total * 100 if total > 0 else 0
        print(f"Pass rate: {pct:.1f}%")

    # Link integrity check
    if args.check_links:
        link_result = validate_links(output_dir)
        n_broken = len(link_result.broken)
        n_external = len(link_result.external)
        print(f"\nLinks: {n_broken} broken, {n_external} external (skipped)")
        if link_result.broken:
            for src, href in link_result.broken:
                print(f"  BROKEN  {src} -> {href}")
            return 1
        if args.verbose and link_result.external:
            for src, href in link_result.external:
                print(f"  external  {src} -> {href}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
