#!/usr/bin/env python3
"""Generate a markdown summary from JUnit XML test results."""

from __future__ import annotations

import argparse
import sys
import textwrap
import xml.etree.ElementTree as ET


def generate_summary(xml_path: str) -> str:
    """Parse JUnit XML and generate markdown summary."""
    lines: list[str] = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        testsuite = root.find("testsuite") if root.tag != "testsuite" else root
        if testsuite is None:
            testsuite = root

        tests = int(testsuite.get("tests", 0))
        failures = int(testsuite.get("failures", 0))
        errors = int(testsuite.get("errors", 0))
        skipped = int(testsuite.get("skipped", 0))
        time_taken = float(testsuite.get("time", 0))

        passed = tests - failures - errors - skipped

        if failures == 0 and errors == 0:
            status = "✓ All tests passed"
        else:
            status = "✗ Some tests failed"

        lines.append("## STT Test Results\n")
        lines.append(f"**Status:** {status}\n")
        lines.append("| Metric | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| ✓ Passed | {passed} |")
        lines.append(f"| ✗ Failed | {failures} |")
        lines.append(f"| × Errors | {errors} |")
        lines.append(f"| → Skipped | {skipped} |")
        lines.append(f"| ▣ Total | {tests} |")
        lines.append(f"| ⏱ Duration | {time_taken:.1f}s |")
        lines.append("")

        if failures > 0 or errors > 0:
            lines.append("<details>")
            lines.append("<summary>Failed Tests</summary>\n")
            for testcase in testsuite.iter("testcase"):
                failure = testcase.find("failure")
                error = testcase.find("error")
                if failure is not None or error is not None:
                    name = testcase.get("name", "unknown")
                    classname = testcase.get("classname", "")
                    elem = failure if failure is not None else error
                    msg = elem.text if elem.text else elem.get("message", "")
                    msg = msg[:2000]
                    indented_msg = textwrap.indent(msg.strip(), "  ")
                    lines.append(f"- **{classname}::{name}**")
                    lines.append("  ```")
                    lines.append(indented_msg)
                    lines.append("  ```")
            lines.append("</details>")

        skipped_tests = [tc for tc in testsuite.iter("testcase") if tc.find("skipped") is not None]
        if skipped_tests:
            lines.append("<details>")
            lines.append("<summary>Skipped Tests</summary>\n")
            lines.append("| Test | Reason |")
            lines.append("|------|--------|")
            for testcase in skipped_tests[:20]:
                name = testcase.get("name", "unknown")
                classname = testcase.get("classname", "")
                skip = testcase.find("skipped")
                reason = skip.get("message", "") if skip is not None else ""
                test_name = f"{classname}::{name}".replace("|", "\\|")
                reason_escaped = reason.strip().replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{test_name}` | {reason_escaped} |")
            if len(skipped_tests) > 20:
                lines.append(f"| ... | ... and {len(skipped_tests) - 20} more |")
            lines.append("</details>")

    except Exception as e:
        lines.append("## STT Test Results\n")
        lines.append(f"⚠ Could not parse test results: {e}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate test summary from JUnit XML")
    parser.add_argument(
        "xml_path",
        nargs="?",
        default="test-results.xml",
        help="Path to JUnit XML file (default: test-results.xml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    summary = generate_summary(args.xml_path)

    if args.output:
        with open(args.output, "w") as f:
            f.write(summary)
    else:
        print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
