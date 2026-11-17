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
    assert check_source('logger.error("failed: %s", event.error)')
    assert check_source('logger.error(f"failed: {event.error.message}")')


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
    assert check_source(f'logger.warning("failed", extra={{"{key}": str(failure)}})')
    assert not check_source(f'logger.warning("failed", extra={{"lk.pii.{key}": str(failure)}})')


def test_exception_class_names_are_safe_metadata() -> None:
    assert not check_source("""
try:
    operation()
except ValueError as failure:
    logger.warning("operation failed (%s)", type(failure).__name__)
""")
