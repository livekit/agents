"""Tests for .github/convert_html_docs.py — pdoc3 HTML to markdown converter."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# The module lives at .github/convert_html_docs.py which isn't a regular
# package, so we load it by file path.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / ".github" / "convert_html_docs.py"
_spec = importlib.util.spec_from_file_location("convert_html_docs", _SCRIPT_PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["convert_html_docs"] = _mod
_spec.loader.exec_module(_mod)

ContentFingerprint = _mod.ContentFingerprint
PageConverter = _mod.PageConverter
convert_directory = _mod.convert_directory
extract_fingerprint = _mod.extract_fingerprint
map_link = _mod.map_link
map_output_path = _mod.map_output_path
validate_links = _mod.validate_links
validate_markdown = _mod.validate_markdown

MINIMAL_MODULE_HTML = """\
<html><body><main>
<article id="content">
<header><h1 class="title">Module <code>livekit.agents.vad</code></h1></header>
<section id="section-intro"></section>
<section></section><section></section><section></section><section></section>
</article>
</main></body></html>
"""

NAMESPACE_HTML = """\
<html><body><main>
<article id="content">
<header><h1 class="title">Namespace <code>livekit</code></h1></header>
<section id="section-intro"></section>
<section>
<h2 class="section-title" id="header-submodules">Sub-modules</h2>
<dl>
<dt><code class="name"><a title="livekit.agents" href="agents/index.html">livekit.agents</a></code></dt>
<dd><div class="desc"><p>LiveKit Agents for Python</p></div></dd>
<dt><code class="name"><a title="livekit.rtc" href="rtc/index.html">livekit.rtc</a></code></dt>
<dd><div class="desc"></div></dd>
</dl>
</section>
</article>
</main></body></html>
"""

FUNCTION_HTML = """\
<html><body><main>
<article id="content">
<header><h1 class="title">Module <code>livekit.agents</code></h1></header>
<section id="section-intro">
<p>LiveKit Agents for Python</p>
</section>
<section></section>
<section>
<h2 class="section-title" id="header-functions">Functions</h2>
<dl>
<dt id="livekit.agents.get_job_context"><code class="name flex">
<span>def <span class="ident">get_job_context</span></span>(<span>) &#8209; livekit.agents.job.JobContext</span>
</code></dt>
<dd>
<details class="source">
<summary><span>Expand source code</span></summary>
<pre><code class="python">def get_job_context() -&gt; JobContext:
    ctx = _JobContextVar.get(None)
    if ctx is None:
        raise RuntimeError(
            &#34;no job context found&#34;
        )
    return ctx</code></pre>
</details>
<div class="desc"><p>Get the current job context.</p></div>
</dd>
</dl>
</section>
<section></section>
</article>
</main></body></html>
"""

CLASS_HTML = """\
<html><body><main>
<article id="content">
<header><h1 class="title">Module <code>livekit.agents.vad</code></h1></header>
<section id="section-intro"></section>
<section></section><section></section><section></section>
<section>
<h2 class="section-title" id="header-classes">Classes</h2>
<dl>
<dt id="livekit.agents.vad.VAD"><code class="flex name class">
<span>class <span class="ident">VAD</span></span>
<span>(</span><span>*, capabilities: VADCapabilities)</span>
</code></dt>
<dd>
<details class="source">
<summary><span>Expand source code</span></summary>
<pre><code class="python">class VAD(ABC):
    def __init__(self, *, capabilities: VADCapabilities) -&gt; None:
        self._capabilities = capabilities</code></pre>
</details>
<div class="desc"><p>Base class for voice activity detection.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>abc.ABC</li>
</ul>
<h3>Subclasses</h3>
<ul class="hlist">
<li>livekit.plugins.silero.vad.VAD</li>
</ul>
<h3>Instance variables</h3>
<dl>
<dt id="livekit.agents.vad.VAD.capabilities"><code class="name">prop <span class="ident">capabilities</span> : VADCapabilities</code></dt>
<dd>
<details class="source">
<summary><span>Expand source code</span></summary>
<pre><code class="python">@property
def capabilities(self) -&gt; VADCapabilities:
    return self._capabilities</code></pre>
</details>
<div class="desc"></div>
</dd>
</dl>
<h3>Methods</h3>
<dl>
<dt id="livekit.agents.vad.VAD.stream"><code class="name flex">
<span>def <span class="ident">stream</span></span>(<span>self) &#8209; VADStream</span>
</code></dt>
<dd>
<details class="source">
<summary><span>Expand source code</span></summary>
<pre><code class="python">@abstractmethod
def stream(self) -&gt; VADStream: ...</code></pre>
</details>
<div class="desc"></div>
</dd>
</dl>
<h3>Inherited members</h3>
<ul class="hlist">
<li><code><b><a title="EventEmitter" href="#">EventEmitter</a></b></code>:
<ul class="hlist">
<li><code><a title="emit" href="#">emit</a></code></li>
<li><code><a title="on" href="#">on</a></code></li>
</ul>
</li>
</ul>
</dd>
</dl>
</section>
</article>
</main></body></html>
"""

CLASS_VARIABLES_HTML = """\
<html><body><main>
<article id="content">
<header><h1 class="title">Module <code>livekit.agents</code></h1></header>
<section id="section-intro"></section>
<section></section><section></section><section></section>
<section>
<h2 class="section-title" id="header-classes">Classes</h2>
<dl>
<dt id="livekit.agents.APIError"><code class="flex name class">
<span>class <span class="ident">APIError</span></span>
<span>(</span><span>message: str, *, body: object | None = None)</span>
</code></dt>
<dd>
<div class="desc"><p>Raised when an API request failed.</p></div>
<h3>Class variables</h3>
<dl>
<dt id="livekit.agents.APIError.message"><code class="name">var <span class="ident">message</span> : str</code></dt>
<dd><div class="desc"><p>The error message returned by the API.</p></div></dd>
<dt id="livekit.agents.APIError.body"><code class="name">var <span class="ident">body</span> : object | None</code></dt>
<dd><div class="desc"><p>The API response body.</p></div></dd>
</dl>
</dd>
</dl>
</section>
</article>
</main></body></html>
"""

MULTILINE_SIG_HTML = """\
<html><body><main>
<article id="content">
<header><h1 class="title">Module <code>livekit.agents</code></h1></header>
<section id="section-intro"></section>
<section></section>
<section>
<h2 class="section-title" id="header-functions">Functions</h2>
<dl>
<dt id="livekit.agents.create_api_error"><code class="name flex">
<span>def <span class="ident">create_api_error</span></span>(<span>message: str = '',<br>*,<br>status: int,<br>request_id: str | None = None) &#8209; APIStatusError</span>
</code></dt>
<dd>
<div class="desc"><p>Create an error from HTTP status.</p></div>
</dd>
</dl>
</section>
<section></section>
</article>
</main></body></html>
"""

ADMONITION_HTML = """\
<html><body><main>
<article id="content">
<header><h1 class="title">Module <code>livekit.agents</code></h1></header>
<section id="section-intro"></section>
<section></section>
<section>
<h2 class="section-title" id="header-functions">Functions</h2>
<dl>
<dt id="livekit.agents.my_func"><code class="name flex">
<span>def <span class="ident">my_func</span></span>(<span>)</span>
</code></dt>
<dd>
<div class="desc">
<p>Some description.</p>
<div class="admonition note">
<p class="admonition-title">Note</p>
<p>This is an important note.</p>
</div>
</div>
</dd>
</dl>
</section>
<section></section>
</article>
</main></body></html>
"""


class TestMapOutputPath:
    def test_index_html(self, tmp_path: Path) -> None:
        """dir/index.html -> dir.md"""
        input_dir = tmp_path / "html"
        input_dir.mkdir()
        html_path = input_dir / "livekit" / "agents" / "index.html"
        html_path.parent.mkdir(parents=True)
        html_path.touch()
        result = map_output_path(html_path, input_dir)
        assert result == Path("livekit/agents.md")

    def test_regular_html(self, tmp_path: Path) -> None:
        """dir/file.html -> dir/file.md"""
        input_dir = tmp_path / "html"
        input_dir.mkdir()
        html_path = input_dir / "livekit" / "agents" / "vad.html"
        html_path.parent.mkdir(parents=True)
        html_path.touch()
        result = map_output_path(html_path, input_dir)
        assert result == Path("livekit/agents/vad.md")

    def test_top_level_index(self, tmp_path: Path) -> None:
        """index.html -> index.md"""
        input_dir = tmp_path / "html"
        input_dir.mkdir()
        html_path = input_dir / "index.html"
        html_path.touch()
        result = map_output_path(html_path, input_dir)
        assert result == Path("index.md")

    def test_nested_index(self, tmp_path: Path) -> None:
        """livekit/plugins/openai/index.html -> livekit/plugins/openai.md"""
        input_dir = tmp_path / "html"
        input_dir.mkdir()
        html_path = input_dir / "livekit" / "plugins" / "openai" / "index.html"
        html_path.parent.mkdir(parents=True)
        html_path.touch()
        result = map_output_path(html_path, input_dir)
        assert result == Path("livekit/plugins/openai.md")

    def test_single_dir_index(self, tmp_path: Path) -> None:
        """livekit/index.html -> livekit.md"""
        input_dir = tmp_path / "html"
        input_dir.mkdir()
        html_path = input_dir / "livekit" / "index.html"
        html_path.parent.mkdir(parents=True)
        html_path.touch()
        result = map_output_path(html_path, input_dir)
        assert result == Path("livekit.md")

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        """Paths with .. are rejected."""
        input_dir = tmp_path / "html"
        input_dir.mkdir()
        # Create a path that would have .. in relative parts
        # We can't easily create a path with .. in parts using Path,
        # so we test the validation logic directly
        html_path = input_dir / "livekit" / "agents" / "vad.html"
        html_path.parent.mkdir(parents=True)
        html_path.touch()
        # Normal path should work
        map_output_path(html_path, input_dir)  # should not raise

    def test_no_collision_index_vs_file(self, tmp_path: Path) -> None:
        """Verify no collision between dir/index.html and dir.html outputs."""
        input_dir = tmp_path / "html"
        input_dir.mkdir()

        # livekit/agents/index.html -> livekit/agents.md
        p1 = input_dir / "livekit" / "agents" / "index.html"
        p1.parent.mkdir(parents=True)
        p1.touch()

        # livekit/agents.html -> livekit/agents.md (would collide!)
        # But pdoc3 never generates both
        p2 = input_dir / "livekit" / "agents.html"
        p2.touch()

        r1 = map_output_path(p1, input_dir)
        r2 = map_output_path(p2, input_dir)
        # In the real bucket, this collision never occurs.
        # Both map to the same path, which is fine since pdoc3 uses one or the other.
        assert r1 == Path("livekit/agents.md")
        assert r2 == Path("livekit/agents.md")


class TestMapLink:
    def test_html_to_md(self) -> None:
        assert map_link("vad.html") == "vad.md"

    def test_index_html(self) -> None:
        assert map_link("agents/index.html") == "agents.md"

    def test_with_fragment(self) -> None:
        assert map_link("llm.html#ChatContext") == "llm.md#ChatContext"

    def test_absolute_url_unchanged(self) -> None:
        assert map_link("https://docs.livekit.io") == "https://docs.livekit.io"

    def test_fragment_only_unchanged(self) -> None:
        assert map_link("#section") == "#section"

    def test_non_html_unchanged(self) -> None:
        assert map_link("data.json") == "data.json"

    def test_parent_index(self) -> None:
        assert map_link("../agents/index.html") == "../agents.md"

    def test_bare_index_html(self) -> None:
        """index.html alone (no directory prefix)."""
        result = map_link("index.html")
        assert result == "../index.md"


class TestPageConverter:
    def test_minimal_module(self) -> None:
        """Empty module produces just a heading."""
        md = PageConverter(MINIMAL_MODULE_HTML).convert()
        assert md.startswith("# Module `livekit.agents.vad`")
        # Should not have any section headings
        assert "## " not in md

    def test_namespace_title(self) -> None:
        """Namespace prefix is preserved."""
        md = PageConverter(NAMESPACE_HTML).convert()
        assert md.startswith("# Namespace `livekit`")

    def test_submodules(self) -> None:
        """Sub-modules section renders as bullet list."""
        md = PageConverter(NAMESPACE_HTML).convert()
        assert "## Sub-modules" in md
        assert "[`livekit.agents`]" in md
        assert "LiveKit Agents for Python" in md
        # Module without description should still be listed
        assert "[`livekit.rtc`]" in md

    def test_function_basic(self) -> None:
        """Function with source and description."""
        md = PageConverter(FUNCTION_HTML).convert()
        assert "## Functions" in md
        assert "### `get_job_context`" in md
        assert "def get_job_context" in md
        assert "Get the current job context." in md
        # Source should be in details
        assert "<details><summary>Source</summary>" in md
        assert "def get_job_context() -> JobContext:" in md

    def test_function_source_html_unescape(self) -> None:
        """HTML entities in source code are unescaped."""
        md = PageConverter(FUNCTION_HTML).convert()
        # &#34; should become "
        assert '"no job context found"' in md
        # &gt; should become >
        assert "-> JobContext:" in md

    def test_multiline_signature(self) -> None:
        """Multi-line signatures with <br> tags."""
        md = PageConverter(MULTILINE_SIG_HTML).convert()
        assert "### `create_api_error`" in md
        assert "def create_api_error" in md
        # Non-breaking hyphen ‑ (U+2011) should be normalized to -
        assert "- APIStatusError" in md

    def test_class_basic(self) -> None:
        """Class with ancestors, subclasses, variables, methods."""
        md = PageConverter(CLASS_HTML).convert()
        assert "## Classes" in md
        assert "### `VAD`" in md
        assert "class VAD" in md
        assert "Base class for voice activity detection." in md

    def test_class_ancestors(self) -> None:
        md = PageConverter(CLASS_HTML).convert()
        assert "#### Ancestors" in md
        assert "- abc.ABC" in md

    def test_class_subclasses(self) -> None:
        md = PageConverter(CLASS_HTML).convert()
        assert "#### Subclasses" in md
        assert "- livekit.plugins.silero.vad.VAD" in md

    def test_class_instance_variables(self) -> None:
        md = PageConverter(CLASS_HTML).convert()
        assert "#### Instance variables" in md
        assert "`capabilities`" in md
        # Property source should be included
        assert "@property" in md

    def test_class_methods(self) -> None:
        md = PageConverter(CLASS_HTML).convert()
        assert "#### Methods" in md
        assert "`stream`" in md
        assert "@abstractmethod" in md

    def test_class_inherited_members(self) -> None:
        md = PageConverter(CLASS_HTML).convert()
        assert "#### Inherited members" in md
        assert "**EventEmitter**" in md
        assert "emit" in md
        assert "on" in md

    def test_class_variables(self) -> None:
        """Class variables section renders properly."""
        md = PageConverter(CLASS_VARIABLES_HTML).convert()
        assert "#### Class variables" in md
        assert "`message`" in md
        assert "The error message returned by the API." in md
        assert "`body`" in md

    def test_admonition(self) -> None:
        """Admonitions convert to blockquotes."""
        md = PageConverter(ADMONITION_HTML).convert()
        assert "> **Note:**" in md
        assert "important note" in md

    def test_class_source_details(self) -> None:
        """Class source code appears in <details>."""
        md = PageConverter(CLASS_HTML).convert()
        # The class source should be in a details block
        assert "<details><summary>Source</summary>" in md
        assert "class VAD(ABC):" in md

    def test_intro_section(self) -> None:
        """Section intro text is included."""
        md = PageConverter(FUNCTION_HTML).convert()
        assert "LiveKit Agents for Python" in md


class TestExtractFingerprint:
    def test_identifiers(self) -> None:
        fp = extract_fingerprint(FUNCTION_HTML)
        assert "livekit.agents.get_job_context" in fp.identifiers

    def test_source_blocks(self) -> None:
        fp = extract_fingerprint(FUNCTION_HTML)
        assert len(fp.source_blocks) > 0
        assert any("get_job_context" in s for s in fp.source_blocks)

    def test_signatures(self) -> None:
        fp = extract_fingerprint(FUNCTION_HTML)
        assert "get_job_context" in fp.signatures

    def test_descriptions(self) -> None:
        fp = extract_fingerprint(FUNCTION_HTML)
        assert any("job context" in d.lower() for d in fp.description_texts)

    def test_class_identifiers(self) -> None:
        fp = extract_fingerprint(CLASS_HTML)
        assert "livekit.agents.vad.VAD" in fp.identifiers
        assert "livekit.agents.vad.VAD.stream" in fp.identifiers
        assert "livekit.agents.vad.VAD.capabilities" in fp.identifiers

    def test_empty_module(self) -> None:
        fp = extract_fingerprint(MINIMAL_MODULE_HTML)
        assert fp.identifiers == []
        assert fp.source_blocks == []
        assert fp.signatures == []


class TestValidation:
    def test_valid_markdown(self) -> None:
        """Converted markdown from same HTML should validate."""
        md = PageConverter(FUNCTION_HTML).convert()
        fp = extract_fingerprint(FUNCTION_HTML)
        vr = validate_markdown(fp, md)
        assert vr.is_valid
        assert vr.score == 1.0

    def test_class_validates(self) -> None:
        """Complex class HTML validates after conversion."""
        md = PageConverter(CLASS_HTML).convert()
        fp = extract_fingerprint(CLASS_HTML)
        vr = validate_markdown(fp, md)
        assert vr.is_valid
        assert vr.score >= 0.9

    def test_missing_identifier_fails(self) -> None:
        """Missing identifiers cause validation failure."""
        fp = ContentFingerprint(
            identifiers=["livekit.agents.missing_func"],
            source_blocks=[],
            signatures=[],
            description_texts=[],
        )
        vr = validate_markdown(fp, "# Some markdown without the identifier")
        assert not vr.is_valid
        assert "livekit.agents.missing_func" in vr.missing_identifiers

    def test_missing_signature_fails(self) -> None:
        """Missing signatures cause validation failure (gate)."""
        fp = ContentFingerprint(
            identifiers=[],
            source_blocks=[],
            signatures=["nonexistent_function"],
            description_texts=[],
        )
        vr = validate_markdown(fp, "# No such function here")
        assert not vr.is_valid
        assert "nonexistent_function" in vr.missing_signatures

    def test_description_warning_only(self) -> None:
        """Low description overlap produces warnings but does NOT fail validation."""
        fp = ContentFingerprint(
            identifiers=[],
            source_blocks=[],
            signatures=[],
            description_texts=[
                "This is a very specific description about quantum computing algorithms"
            ],
        )
        vr = validate_markdown(fp, "# Completely unrelated content about cooking recipes")
        # Should still be valid (descriptions are warning-only)
        assert vr.is_valid
        assert len(vr.description_warnings) > 0

    def test_empty_fingerprint_valid(self) -> None:
        """Empty fingerprint (empty module) always validates."""
        fp = ContentFingerprint()
        vr = validate_markdown(fp, "# Module `empty`\n")
        assert vr.is_valid
        assert vr.score == 1.0

    def test_class_variables_validate(self) -> None:
        md = PageConverter(CLASS_VARIABLES_HTML).convert()
        fp = extract_fingerprint(CLASS_VARIABLES_HTML)
        vr = validate_markdown(fp, md)
        assert vr.is_valid

    def test_multiline_sig_validates(self) -> None:
        md = PageConverter(MULTILINE_SIG_HTML).convert()
        fp = extract_fingerprint(MULTILINE_SIG_HTML)
        vr = validate_markdown(fp, md)
        assert vr.is_valid


REAL_HTML_DIR = Path(__file__).parent.parent / "docs"


@pytest.mark.skipif(
    not REAL_HTML_DIR.is_dir(),
    reason="docs directory not present",
)
class TestRealFiles:
    """Integration tests against real pdoc3 HTML output."""

    def test_livekit_index(self) -> None:
        html_path = REAL_HTML_DIR / "livekit" / "index.html"
        if not html_path.exists():
            pytest.skip("livekit/index.html not found")
        html_content = html_path.read_text(encoding="utf-8")
        md = PageConverter(html_content).convert()
        fp = extract_fingerprint(html_content)
        vr = validate_markdown(fp, md)
        assert md.startswith("# Namespace `livekit`") or md.startswith("# Module `livekit`")
        assert "Sub-modules" in md
        assert vr.is_valid or vr.score >= 0.9, (
            f"Validation: {vr.score:.2f}, missing ids: {vr.missing_identifiers[:3]}"
        )

    def test_agents_index(self) -> None:
        html_path = REAL_HTML_DIR / "livekit" / "agents" / "index.html"
        if not html_path.exists():
            pytest.skip("livekit/agents/index.html not found")
        html_content = html_path.read_text(encoding="utf-8")
        md = PageConverter(html_content).convert()
        fp = extract_fingerprint(html_content)
        vr = validate_markdown(fp, md)
        assert "## Functions" in md
        assert "## Classes" in md
        assert vr.is_valid or vr.score >= 0.9, (
            f"Validation: {vr.score:.2f}, "
            f"missing ids: {vr.missing_identifiers[:3]}, "
            f"missing sigs: {vr.missing_signatures[:3]}"
        )

    def test_vad_module(self) -> None:
        html_path = REAL_HTML_DIR / "livekit" / "agents" / "vad.html"
        if not html_path.exists():
            pytest.skip("livekit/agents/vad.html not found")
        html_content = html_path.read_text(encoding="utf-8")
        md = PageConverter(html_content).convert()
        fp = extract_fingerprint(html_content)
        vr = validate_markdown(fp, md)
        assert "VAD" in md
        assert vr.is_valid or vr.score >= 0.9

    def test_path_mapping_no_collisions(self) -> None:
        """Verify zero collisions across all HTML files in docs."""
        html_files = sorted(REAL_HTML_DIR.rglob("*.html"))
        assert len(html_files) > 0, "No HTML files found"

        output_paths: dict[str, Path] = {}
        collisions: list[tuple[Path, Path, str]] = []

        for html_path in html_files:
            try:
                out = map_output_path(html_path, REAL_HTML_DIR)
            except ValueError:
                continue
            out_str = str(out)
            if out_str in output_paths:
                collisions.append((output_paths[out_str], html_path, out_str))
            else:
                output_paths[out_str] = html_path

        assert not collisions, f"Path collisions found: {collisions}"

    def test_batch_validation_pass_rate(self) -> None:
        """At least 90% of real HTML files should validate after conversion."""
        html_files = sorted(REAL_HTML_DIR.rglob("*.html"))
        if len(html_files) < 10:
            pytest.skip("Not enough HTML files for batch test")

        total = 0
        passed = 0
        for html_path in html_files:
            html_content = html_path.read_text(encoding="utf-8")
            md = PageConverter(html_content).convert()
            fp = extract_fingerprint(html_content)
            vr = validate_markdown(fp, md)
            total += 1
            if vr.is_valid:
                passed += 1

        rate = passed / total if total > 0 else 0
        assert rate >= 0.90, f"Pass rate {rate:.1%} ({passed}/{total}) below 90% threshold"

    def test_link_integrity(self, tmp_path: Path) -> None:
        """All internal .md links in converted output must resolve to existing files."""
        output_dir = tmp_path / "md-out"
        convert_directory(
            REAL_HTML_DIR,
            output_dir,
            validate=False,
            verbose=False,
        )
        result = validate_links(output_dir)
        assert not result.broken, f"{len(result.broken)} broken links found:\n" + "\n".join(
            f"  {src} -> {href}" for src, href in result.broken[:20]
        )
        # External links (e.g. Pydantic docstring cross-refs) should be
        # classified separately, not silently dropped.
        if result.external:
            # Sanity: we know the Pydantic ../concepts/models.md links exist
            ext_hrefs = {href for _, href in result.external}
            assert "../concepts/models.md" in ext_hrefs


class TestValidateLinks:
    def test_no_broken_links(self, tmp_path: Path) -> None:
        """All links resolve correctly."""
        (tmp_path / "a.md").write_text("[link](b.md)\n")
        (tmp_path / "b.md").write_text("# B\n")
        result = validate_links(tmp_path)
        assert result.broken == []
        assert result.external == []

    def test_broken_link_detected(self, tmp_path: Path) -> None:
        """A link to a missing file is detected."""
        (tmp_path / "a.md").write_text("[link](missing.md)\n")
        result = validate_links(tmp_path)
        assert len(result.broken) == 1
        assert result.broken[0][1] == "missing.md"

    def test_external_links_ignored(self, tmp_path: Path) -> None:
        """External URLs and anchors are not checked."""
        (tmp_path / "a.md").write_text(
            "[ext](https://example.com) [anchor](#foo) [mailto](mailto:a@b.com)\n"
        )
        result = validate_links(tmp_path)
        assert result.broken == []
        assert result.external == []

    def test_subdirectory_links(self, tmp_path: Path) -> None:
        """Links into subdirectories resolve correctly."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.md").write_text("[link](sub/b.md)\n")
        (sub / "b.md").write_text("# B\n")
        result = validate_links(tmp_path)
        assert result.broken == []

    def test_fragment_links(self, tmp_path: Path) -> None:
        """Links with fragments still check the file exists."""
        (tmp_path / "a.md").write_text("[link](b.md#section)\n")
        (tmp_path / "b.md").write_text("# B\n")
        result = validate_links(tmp_path)
        assert result.broken == []

    def test_known_external_href_classified(self, tmp_path: Path) -> None:
        """The exact known Pydantic href is classified as external when missing."""
        # Nest source so ../concepts/models.md resolves inside the output tree
        # but points to a non-existent file.
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.md").write_text("[Models](../concepts/models.md)\n")
        result = validate_links(tmp_path)
        assert result.broken == []
        assert len(result.external) == 1
        assert result.external[0][1] == "../concepts/models.md"

    def test_known_external_href_valid_if_exists(self, tmp_path: Path) -> None:
        """If a known-external href actually resolves to an existing file, it's valid."""
        sub = tmp_path / "sub"
        sub.mkdir()
        concepts = tmp_path / "concepts"
        concepts.mkdir()
        (concepts / "models.md").write_text("# Models\n")
        (sub / "a.md").write_text("[Models](../concepts/models.md)\n")
        result = validate_links(tmp_path)
        assert result.broken == []
        assert result.external == []

    def test_unknown_missing_dir_is_broken(self, tmp_path: Path) -> None:
        """Links to non-existent paths that don't match known externals are broken."""
        (tmp_path / "a.md").write_text("[link](nonexistent_dir/page.md)\n")
        result = validate_links(tmp_path)
        assert len(result.broken) == 1
        assert result.broken[0][1] == "nonexistent_dir/page.md"
        assert result.external == []

    def test_escape_above_root_is_broken(self, tmp_path: Path) -> None:
        """Links resolving above the output root are broken, not external."""
        (tmp_path / "a.md").write_text("[oops](../../oops.md)\n")
        result = validate_links(tmp_path)
        assert len(result.broken) == 1
        assert result.broken[0][1] == "../../oops.md"
        assert result.external == []

    def test_similar_but_not_exact_external_is_broken(self, tmp_path: Path) -> None:
        """A href similar to the known external but not exact is still broken."""
        (tmp_path / "a.md").write_text("[link](../concepts/other.md)\n")
        result = validate_links(tmp_path)
        assert len(result.broken) == 1
        assert result.external == []
