class Comparable:
    """Customise how your Enum acts when compared to other objects.

    Your Enum must implement a ``_cmp_values`` method which takes the Enum
    member's value and the other value and manipulates them into the actual
    values that can be compared.

    A case-insensitive StrEnum might look like this::

        class HttpHeader(Comparable, KebabCaseStrEnum):
            ContentType = auto()
            Host = auto()
            Accept = auto()
            XForwardedFor = auto()

            def _cmp_values(self, other):
                return self.value.lower(), str(other).lower()

    You could then use these headers in case-insensitive comparisons::

        assert "Content-Type" == HttpHeader.ContentType
        assert "content-type" == HttpHeader.ContentType
        assert "coNtEnt-tyPe" == HttpHeader.ContentType

    .. note::
        Your ``_cmp_values`` method *must not* return ``self`` as one of the
        values to be compared -- that would result in infinite recursion.
        Instead, perform operations on ``self.value`` and return that.

    .. warning::
        A bug in Python prior to 3.7.1 prevents mix-ins working with Enum
        subclasses.

    .. versionadded:: 0.4.6
    """

    def __eq__(self, other):
        value, other = self._cmp_values(other)
        return value == other

    def __ne__(self, other):
        value, other = self._cmp_values(other)
        return value != other

    def __lt__(self, other):
        value, other = self._cmp_values(other)
        return value < other

    def __le__(self, other):
        value, other = self._cmp_values(other)
        return value <= other

    def __gt__(self, other):
        value, other = self._cmp_values(other)
        return value > other

    def __ge__(self, other):
        value, other = self._cmp_values(other)
        return value >= other

    def _cmp_values(self, other):
        raise NotImplementedError(
            "Enum's using Comparable must implement their own _cmp_values function."
        )
