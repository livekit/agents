import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.generate_test_summary import generate_summary

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("tests", "failures", "errors", "skipped", "status"),
    [
        (2, 0, 0, 2, "⚠ All tests skipped"),
        (0, 0, 0, 0, "⚠ No tests ran"),
        (2, 0, 0, 1, "✓ All executed tests passed"),
        (2, 0, 0, 0, "✓ All tests passed"),
        (2, 1, 0, 1, "✗ Some tests failed"),
        (2, 0, 1, 1, "✗ Some tests failed"),
    ],
)
def test_summary_reports_test_outcome(
    tmp_path: Path, tests: int, failures: int, errors: int, skipped: int, status: str
) -> None:
    xml_path = tmp_path / "test-results.xml"
    root = ET.Element("testsuites")
    ET.SubElement(
        root,
        "testsuite",
        tests=str(tests),
        failures=str(failures),
        errors=str(errors),
        skipped=str(skipped),
    )
    ET.ElementTree(root).write(xml_path)

    assert f"**Status:** {status}" in generate_summary(str(xml_path)).splitlines()
