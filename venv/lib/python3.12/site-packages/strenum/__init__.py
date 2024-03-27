import enum
from ._version import get_versions
from ._name_mangler import _NameMangler

__version__ = get_versions()["version"]
__version_info__ = tuple(int(n) for n in __version__.partition("+")[0].split("."))
del get_versions

_name_mangler = _NameMangler()

# The first argument to the `_generate_next_value_` function of the `enum.Enum`
# class is documented to be the name of the enum member, not the enum class:
#
#     https://docs.python.org/3.6/library/enum.html#using-automatic-values
#
# Pylint, though, doesn't know about this so we need to disable it's check for
# `self` arguments.
# pylint: disable=no-self-argument


class StrEnum(str, enum.Enum):
    """
    StrEnum is a Python ``enum.Enum`` that inherits from ``str``. The default
    ``auto()`` behavior uses the member name as its value.

    Example usage::

        class Example(StrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "UPPER_CASE"
        assert Example.lower_case == "lower_case"
        assert Example.MixedCase == "MixedCase"
    """

    def __new__(cls, value, *args, **kwargs):
        if not isinstance(value, (str, enum.auto)):
            raise TypeError(
                f"Values of StrEnums must be strings: {value!r} is a {type(value)}"
            )
        return super().__new__(cls, value, *args, **kwargs)

    def __str__(self):
        return str(self.value)

    def _generate_next_value_(name, *_):
        return name


class LowercaseStrEnum(StrEnum):
    """
    A ``StrEnum`` where ``auto()`` will convert the name to `lowercase` to
    produce each member's value.

    Example usage::

        class Example(LowercaseStrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "upper_case"
        assert Example.lower_case == "lower_case"
        assert Example.MixedCase == "mixedcase"

    .. versionadded:: 0.4.3
    """

    def _generate_next_value_(name, *_):
        return name.lower()


class UppercaseStrEnum(StrEnum):
    """
    A ``StrEnum`` where ``auto()`` will convert the name to `UPPERCASE` to
    produce each member's value.

    Example usage::

        class Example(UppercaseStrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "UPPER_CASE"
        assert Example.lower_case == "LOWER_CASE"
        assert Example.MixedCase == "MIXEDCASE"

    .. versionadded:: 0.4.3
    """

    def _generate_next_value_(name, *_):
        return name.upper()


class CamelCaseStrEnum(StrEnum):
    """
    A ``StrEnum`` where ``auto()`` will convert the name to `camelCase` to
    produce each member's value.

    Example usage::

        class Example(CamelCaseStrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "upperCase"
        assert Example.lower_case == "lowerCase"
        assert Example.MixedCase == "mixedCase"

    .. versionadded:: 0.4.5
    """

    def _generate_next_value_(name, *_):
        return _name_mangler.camel(name)


class PascalCaseStrEnum(StrEnum):
    """
    A ``StrEnum`` where ``auto()`` will convert the name to `PascalCase` to
    produce each member's value.

    Example usage::

        class Example(PascalCaseStrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "UpperCase"
        assert Example.lower_case == "LowerCase"
        assert Example.MixedCase == "MixedCase"

    .. versionadded:: 0.4.5
    """

    def _generate_next_value_(name, *_):
        return _name_mangler.pascal(name)


class KebabCaseStrEnum(StrEnum):
    """
    A ``StrEnum`` where ``auto()`` will convert the name to `kebab-case` to
    produce each member's value.

    Example usage::

        class Example(KebabCaseStrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "upper-case"
        assert Example.lower_case == "lower-case"
        assert Example.MixedCase == "mixed-case"

    .. versionadded:: 0.4.5
    """

    def _generate_next_value_(name, *_):
        return _name_mangler.kebab(name)


class SnakeCaseStrEnum(StrEnum):
    """
    A ``StrEnum`` where ``auto()`` will convert the name to `snake_case` to
    produce each member's value.

    Example usage::

        class Example(SnakeCaseStrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "upper_case"
        assert Example.lower_case == "lower_case"
        assert Example.MixedCase == "mixed_case"

    .. versionadded:: 0.4.5
    """

    def _generate_next_value_(name, *_):
        return _name_mangler.snake(name)


class MacroCaseStrEnum(StrEnum):
    """
    A ``StrEnum`` where ``auto()`` will convert the name to `MACRO_CASE` to
    produce each member's value.

    Example usage::

        class Example(MacroCaseStrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "UPPER_CASE"
        assert Example.lower_case == "LOWER_CASE"
        assert Example.MixedCase == "MIXED_CASE"

    .. versionadded:: 0.4.6
    """

    def _generate_next_value_(name, *_):
        return _name_mangler.macro(name)


class CamelSnakeCaseStrEnum(StrEnum):
    """
    A ``StrEnum`` where ``auto()`` will convert the name to `camel_Snake_Case` to
    produce each member's value.

    Example usage::

        class Example(CamelSnakeCaseStrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "upper_Case"
        assert Example.lower_case == "lower_Case"
        assert Example.MixedCase == "mixed_Case"

    .. versionadded:: 0.4.8
    """

    def _generate_next_value_(name, *_):
        return _name_mangler.camel_snake(name)


class PascalSnakeCaseStrEnum(StrEnum):
    """
    A ``StrEnum`` where ``auto()`` will convert the name to `Pascal_Snake_Case` to
    produce each member's value.

    Example usage::

        class Example(PascalSnakeCaseStrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "Upper_Case"
        assert Example.lower_case == "Lower_Case"
        assert Example.MixedCase == "Mixed_Case"

    .. versionadded:: 0.4.8
    """

    def _generate_next_value_(name, *_):
        return _name_mangler.pascal_snake(name)


class SpongebobCaseStrEnum(StrEnum):
    """
    A ``StrEnum`` where ``auto()`` will convert the name to `SpONGEBob_CAse` to
    produce each member's value.

    Example usage::

        class Example(SpongebobCaseStrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "uPpER_cAsE"
        assert Example.lower_case == "lowER_CASe"
        assert Example.MixedCase == "MixeD_CAse"

    .. versionadded:: 0.4.8
    """

    def _generate_next_value_(name, *_):
        return _name_mangler.spongebob(name)


class CobolCaseStrEnum(StrEnum):
    """
    A ``StrEnum`` where ``auto()`` will convert the name to `COBOL-CASE` to
    produce each member's value.

    Example usage::

        class Example(CobolCaseStrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "UPPER-CASE"
        assert Example.lower_case == "LOWER-CASE"
        assert Example.MixedCase == "MIXED-CASE"

    .. versionadded:: 0.4.8
    """

    def _generate_next_value_(name, *_):
        return _name_mangler.cobol(name)


class HttpHeaderCaseStrEnum(StrEnum):
    """
    A ``StrEnum`` where ``auto()`` will convert the name to `Http-Header-Case` to
    produce each member's value.

    Example usage::

        class Example(HttpHeaderCaseStrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()

        assert Example.UPPER_CASE == "Upper-Case"
        assert Example.lower_case == "Lower-Case"
        assert Example.MixedCase == "Mixed-Case"

    .. versionadded:: 0.4.8
    """

    def _generate_next_value_(name, *_):
        return _name_mangler.http_header(name)
