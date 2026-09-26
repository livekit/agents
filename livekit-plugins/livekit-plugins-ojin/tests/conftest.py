import pathlib
import sys

# The repository's root `tests` package is imported as a pytest plugin, so this
# directory's own package name resolves there instead. Make the shared helpers
# importable by module name rather than relatively.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
