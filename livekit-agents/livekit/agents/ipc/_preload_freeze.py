"""Freeze forkserver preloads to preserve copy-on-write sharing.

A full cyclic collection in a forked job process would otherwise write to GC
metadata on inherited objects and make those memory pages private. This module
must remain last in the AgentServer forkserver preload list.
"""

import gc

# Collect unreachable cycles before they become permanent in the forkserver.
gc.collect()
gc.freeze()
