"""``python -m livekit.agents``: the subprocess interface ``lk agent`` drives (``cli.main``)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
