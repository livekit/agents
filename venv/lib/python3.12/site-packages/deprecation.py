# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
import collections
import functools
import textwrap
import warnings

from packaging import version
from datetime import date

__version__ = "2.1.0"

# This is mostly here so automodule docs are ordered more ideally.
__all__ = ["deprecated", "message_location", "fail_if_not_removed",
           "DeprecatedWarning", "UnsupportedWarning"]

#: Location where the details are added to a deprecated docstring
#:
#: When set to ``"bottom"``, the details are appended to the end.
#: When set to ``"top"``, the details are inserted between the
#: summary line and docstring contents.
message_location = "bottom"


class DeprecatedWarning(DeprecationWarning):
    """A warning class for deprecated methods

    This is a specialization of the built-in :class:`DeprecationWarning`,
    adding parameters that allow us to get information into the __str__
    that ends up being sent through the :mod:`warnings` system.
    The attributes aren't able to be retrieved after the warning gets
    raised and passed through the system as only the class--not the
    instance--and message are what gets preserved.

    :param function: The function being deprecated.
    :param deprecated_in: The version that ``function`` is deprecated in
    :param removed_in: The version or :class:`datetime.date` specifying
                       when ``function`` gets removed.
    :param details: Optional details about the deprecation. Most often
                    this will include directions on what to use instead
                    of the now deprecated code.
    """

    def __init__(self, function, deprecated_in, removed_in, details=""):
        # NOTE: The docstring only works for this class if it appears up
        # near the class name, not here inside __init__. I think it has
        # to do with being an exception class.
        self.function = function
        self.deprecated_in = deprecated_in
        self.removed_in = removed_in
        self.details = details
        super(DeprecatedWarning, self).__init__(function, deprecated_in,
                                                removed_in, details)

    def __str__(self):
        # Use a defaultdict to give us the empty string
        # when a part isn't included.
        parts = collections.defaultdict(str)
        parts["function"] = self.function

        if self.deprecated_in:
            parts["deprecated"] = " as of %s" % self.deprecated_in
        if self.removed_in:
            parts["removed"] = " and will be removed {} {}".format("on" if isinstance(self.removed_in, date) else "in",
                                                                   self.removed_in)
        if any([self.deprecated_in, self.removed_in, self.details]):
            parts["period"] = "."
        if self.details:
            parts["details"] = " %s" % self.details

        return ("%(function)s is deprecated%(deprecated)s%(removed)s"
                "%(period)s%(details)s" % (parts))


class UnsupportedWarning(DeprecatedWarning):
    """A warning class for methods to be removed

    This is a subclass of :class:`~deprecation.DeprecatedWarning` and is used
    to output a proper message about a function being unsupported.
    Additionally, the :func:`~deprecation.fail_if_not_removed` decorator
    will handle this warning and cause any tests to fail if the system
    under test uses code that raises this warning.
    """

    def __str__(self):
        parts = collections.defaultdict(str)
        parts["function"] = self.function
        parts["removed"] = self.removed_in

        if self.details:
            parts["details"] = " %s" % self.details

        return ("%(function)s is unsupported as of %(removed)s."
                "%(details)s" % (parts))


def deprecated(deprecated_in=None, removed_in=None, current_version=None,
               details=""):
    """Decorate a function to signify its deprecation

    This function wraps a method that will soon be removed and does two things:
        * The docstring of the method will be modified to include a notice
          about deprecation, e.g., "Deprecated since 0.9.11. Use foo instead."
        * Raises a :class:`~deprecation.DeprecatedWarning`
          via the :mod:`warnings` module, which is a subclass of the built-in
          :class:`DeprecationWarning`. Note that built-in
          :class:`DeprecationWarning`s are ignored by default, so for users
          to be informed of said warnings they will need to enable them--see
          the :mod:`warnings` module documentation for more details.

    :param deprecated_in: The version at which the decorated method is
                          considered deprecated. This will usually be the
                          next version to be released when the decorator is
                          added. The default is **None**, which effectively
                          means immediate deprecation. If this is not
                          specified, then the `removed_in` and
                          `current_version` arguments are ignored.
    :param removed_in: The version or :class:`datetime.date` when the decorated
                       method will be removed. The default is **None**,
                       specifying that the function is not currently planned
                       to be removed.
                       Note: This parameter cannot be set to a value if
                       `deprecated_in=None`.
    :param current_version: The source of version information for the
                            currently running code. This will usually be
                            a `__version__` attribute on your library.
                            The default is `None`.
                            When `current_version=None` the automation to
                            determine if the wrapped function is actually
                            in a period of deprecation or time for removal
                            does not work, causing a
                            :class:`~deprecation.DeprecatedWarning`
                            to be raised in all cases.
    :param details: Extra details to be added to the method docstring and
                    warning. For example, the details may point users to
                    a replacement method, such as "Use the foo_bar
                    method instead". By default there are no details.
    """
    # You can't just jump to removal. It's weird, unfair, and also makes
    # building up the docstring weird.
    if deprecated_in is None and removed_in is not None:
        raise TypeError("Cannot set removed_in to a value "
                        "without also setting deprecated_in")

    # Only warn when it's appropriate. There may be cases when it makes sense
    # to add this decorator before a formal deprecation period begins.
    # In CPython, PendingDeprecatedWarning gets used in that period,
    # so perhaps mimick that at some point.
    is_deprecated = False
    is_unsupported = False

    # StrictVersion won't take a None or a "", so make whatever goes to it
    # is at least *something*. Compare versions only if removed_in is not
    # of type datetime.date
    if isinstance(removed_in, date):
        if date.today() >= removed_in:
            is_unsupported = True
        else:
            is_deprecated = True
    elif current_version:
        current_version = version.parse(current_version)

        if (removed_in
                and current_version >= version.parse(removed_in)):
            is_unsupported = True
        elif (deprecated_in
              and current_version >= version.parse(deprecated_in)):
            is_deprecated = True
    else:
        # If we can't actually calculate that we're in a period of
        # deprecation...well, they used the decorator, so it's deprecated.
        # This will cover the case of someone just using
        # @deprecated("1.0") without the other advantages.
        is_deprecated = True

    should_warn = any([is_deprecated, is_unsupported])

    def _function_wrapper(function):
        if should_warn:
            # Everything *should* have a docstring, but just in case...
            existing_docstring = function.__doc__ or ""

            # The various parts of this decorator being optional makes for
            # a number of ways the deprecation notice could go. The following
            # makes for a nicely constructed sentence with or without any
            # of the parts.

            # If removed_in is a date, use "removed on"
            # If removed_in is a version, use "removed in"
            parts = {
                "deprecated_in":
                    " %s" % deprecated_in if deprecated_in else "",
                "removed_in":
                    "\n   This will be removed {} {}.".format("on" if isinstance(removed_in, date) else "in",
                                                              removed_in) if removed_in else "",
                "details":
                    " %s" % details if details else ""}

            deprecation_note = (".. deprecated::{deprecated_in}"
                                "{removed_in}{details}".format(**parts))

            # default location for insertion of deprecation note
            loc = 1

            # split docstring at first occurrence of newline
            string_list = existing_docstring.split("\n", 1)

            if len(string_list) > 1:
                # With a multi-line docstring, when we modify
                # existing_docstring to add our deprecation_note,
                # if we're not careful we'll interfere with the
                # indentation levels of the contents below the
                # first line, or as PEP 257 calls it, the summary
                # line. Since the summary line can start on the
                # same line as the """, dedenting the whole thing
                # won't help. Split the summary and contents up,
                # dedent the contents independently, then join
                # summary, dedent'ed contents, and our
                # deprecation_note.

                # in-place dedent docstring content
                string_list[1] = textwrap.dedent(string_list[1])

                # we need another newline
                string_list.insert(loc, "\n")

                # change the message_location if we add to end of docstring
                # do this always if not "top"
                if message_location != "top":
                    loc = 3

            # insert deprecation note and dual newline
            string_list.insert(loc, deprecation_note)
            string_list.insert(loc, "\n\n")

            function.__doc__ = "".join(string_list)

        @functools.wraps(function)
        def _inner(*args, **kwargs):
            if should_warn:
                if is_unsupported:
                    cls = UnsupportedWarning
                else:
                    cls = DeprecatedWarning

                the_warning = cls(function.__name__, deprecated_in,
                                  removed_in, details)
                warnings.warn(the_warning, category=DeprecationWarning,
                              stacklevel=2)

            return function(*args, **kwargs)
        return _inner
    return _function_wrapper


def fail_if_not_removed(method):
    """Decorate a test method to track removal of deprecated code

    This decorator catches :class:`~deprecation.UnsupportedWarning`
    warnings that occur during testing and causes unittests to fail,
    making it easier to keep track of when code should be removed.

    :raises: :class:`AssertionError` if an
             :class:`~deprecation.UnsupportedWarning`
             is raised while running the test method.
    """
    # NOTE(briancurtin): Unless this is named test_inner, nose won't work
    # properly. See Issue #32.
    @functools.wraps(method)
    def test_inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            rv = method(*args, **kwargs)

        for warning in caught_warnings:
            if warning.category == UnsupportedWarning:
                raise AssertionError(
                    ("%s uses a function that should be removed: %s" %
                     (method, str(warning.message))))
        return rv
    return test_inner
