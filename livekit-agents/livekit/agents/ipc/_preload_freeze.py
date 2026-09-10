"""Preserve copy-on-write sharing for forkserver-preloaded objects.

Cyclic GC in a job process writes inherited object metadata and makes those
pages private.
"""

import gc

# Do not freeze unreachable cycles for the forkserver lifetime.
gc.collect()
gc.freeze()
