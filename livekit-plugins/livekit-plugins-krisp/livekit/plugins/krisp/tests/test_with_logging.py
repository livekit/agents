#!/usr/bin/env python3
"""Test with logging enabled to debug silent output issue."""

import logging

from test_audio_filtering import main

# Enable INFO level logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Now run the test
main()
