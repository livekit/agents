# pylint: disable=no-name-in-module
import re
from zlib import crc32


class _NameMangler:
    _regex = re.compile(r"([A-Z]?[a-z]+)|([A-Z]+(?![a-z]))")

    def words(self, name):
        """
        Split a string into words. Should correctly handle splitting:
            camelCase
            PascalCase
            kebab-case
            snake_case
            MACRO_CASE
            camel_Snake_Case
            Pascal_Snake_Case
            COBOL-CASE
            Http-Header-Case

        It _does not_ handle splitting spongebob case.
        """
        yield from (m.group(0) for m in self._regex.finditer(name))

    def camel(self, name):
        """
        Convert a name to camelCase
        """

        def cased_words(word_iter):
            yield next(word_iter, "").lower()
            yield from (w.title() for w in word_iter)

        return "".join(cased_words(self.words(name)))

    def pascal(self, name):
        """
        Convert a name to PascalCase
        """

        return "".join(w.title() for w in self.words(name))

    def kebab(self, name):
        """
        Convert a name to kebab-case
        """

        return "-".join(w.lower() for w in self.words(name))

    def snake(self, name):
        """
        Convert a name to snake_case
        """

        return "_".join(w.lower() for w in self.words(name))

    def macro(self, name):
        """
        Convert a name to MACRO_CASE
        """

        return "_".join(w.upper() for w in self.words(name))

    # The following are inspired by examples in the Wikipedia
    # [Naming convention](https://en.wikipedia.org/wiki/Naming_convention_(programming))
    # article

    def camel_snake(self, name):
        """
        Convert a name to camel_Snake_Case
        """

        def cased_words(word_iter):
            yield next(word_iter, "").lower()
            yield from (w.title() for w in word_iter)

        return "_".join(cased_words(self.words(name)))

    def pascal_snake(self, name):
        """
        Convert a name to Pascal_Snake_Case
        """

        return "_".join(w.title() for w in self.words(name))

    def spongebob(self, name):
        """
        Convert a name to SpOngEBOb_CASe

        The PRNG we use is seeded with the word to be scrambled. This produces
        stable output so the same input will always produce in the same output.
        It's not `truly` random, but your tests will thank me.
        """

        def prng(seed_word):
            state = 1 << 31 | crc32(seed_word.encode("utf-8")) | 1

            def step(state):
                state = state >> 1 | (state & 0x01 ^ ((state & 0x02) >> 1)) << 31
                bit = state & 0x1
                return bit, state

            for _ in range(100):
                _, state = step(state)
            while True:
                bit, state = step(state)
                yield str.upper if bit else str.lower

        def scramble(word):
            return "".join(f(ch) for ch, f in zip(word, prng(word)))

        return "_".join(scramble(w) for w in self.words(name))

    def cobol(self, name):
        """
        Convert a name to COBOL-CASE
        """

        return "-".join(w.upper() for w in self.words(name))

    def http_header(self, name):
        """
        Convert a name to Http-Header-Case
        """

        return "-".join(w.title() for w in self.words(name))
