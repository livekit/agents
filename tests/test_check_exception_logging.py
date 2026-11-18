from __future__ import annotations

import pytest

from scripts.check_exception_logging import check_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "statement",
    [
        'logger.warning(f"failed: {failure}")',
        'logger.error("failed: %s", failure)',
        'logger.exception("failed: %s", str(failure))',
        'logger.log(30, "failed: %s", failure.message)',
        'logger.error("failed: {}".format(failure))',
        'alias = failure\n    logger.error("failed: %s", alias)',
        'message = f"failed: {failure}"\n    logger.error(message)',
    ],
)
def test_caught_exceptions_cannot_be_formatted_into_log_messages(statement: str) -> None:
    source = f"try:\n    operation()\nexcept ValueError as failure:\n    {statement}\n"
    assert check_source(source)


def test_exception_parameters_and_task_exceptions_are_checked() -> None:
    assert check_source('def failed(*, reason: BaseException):\n    logger.warning("%s", reason)')
    assert check_source('failure = task.exception()\nlogger.warning("%s", failure)')
    assert check_source('failure = APIError("secret")\nlogger.warning("%s", failure.message)')
    assert check_source(
        'def failed(error: APIError):\n    logger.error(f"failed: {error.message}")'
    )


def test_structured_exceptions_and_safe_substitutions_are_allowed() -> None:
    assert not check_source("""
try:
    operation()
except ValueError as failure:
    logger.warning("operation failed, retrying in %ss", interval, exc_info=failure)
    logger.exception("operation failed", extra={"lk.pii.error": str(failure)})
""")


def test_reassigned_values_and_shadowed_parameters_are_not_exceptions() -> None:
    assert not check_source("""
try:
    operation()
except ValueError as failure:
    alias = failure
    alias = "retrying"
    logger.info("%s", alias)
    def status(failure):
        logger.info("%s", failure)
""")


@pytest.mark.parametrize("key", ["error", "err", "errors", "exception"])
def test_exception_extra_fields_must_be_tagged_for_redaction(key: str) -> None:
    assert check_source(
        f"try:\n    operation()\nexcept ValueError as failure:\n"
        f'    logger.warning("failed", extra={{"{key}": str(failure)}})'
    )
    assert not check_source(f'logger.warning("failed", extra={{"lk.pii.{key}": str(failure)}})')


def test_field_names_do_not_establish_sensitivity() -> None:
    assert not check_source(
        'logger.warning("failed", extra={"error": "not supported", "code": 404})'
    )
    assert not check_source('logger.warning("%s", status.message)')
    assert not check_source("""
try:
    operation()
except APIStatusError as failure:
    logger.warning("failed", extra={"error_message": failure.message, "code": failure.status_code})
""")


def test_whole_exception_fields_are_checked_regardless_of_key() -> None:
    assert check_source("""
try:
    operation()
except APIStatusError as failure:
    logger.warning("failed", extra={"details": str(failure)})
""")


def test_exception_class_names_are_safe_metadata() -> None:
    assert not check_source("""
try:
    operation()
except ValueError as failure:
    logger.warning("operation failed (%s)", type(failure).__name__)
""")


def test_selected_field_aliases_are_distinct_from_whole_exception_aliases() -> None:
    assert not check_source("""
try:
    operation()
except APIStatusError as failure:
    diagnostic = failure.message
    logger.warning("failed", extra={"error_message": diagnostic})
""")
    assert check_source("""
try:
    operation()
except APIStatusError as failure:
    diagnostic = str(failure)
    logger.warning("failed", extra={"details": diagnostic})
""")
