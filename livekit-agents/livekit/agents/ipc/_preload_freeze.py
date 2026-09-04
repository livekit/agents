"""Side-effect module: imported **last** in the forkserver preload list.

A forked job process shares the forkserver's pages copy-on-write, but the first
gen2 collection in the child traverses every tracked object and rewrites its GC
header, dirtying (and so un-sharing) nearly every page holding preloaded
objects. ``gc.freeze()`` moves them to the permanent generation, which the
collector never visits, so those pages stay shared for the life of the job.

Kept free of imports on purpose: the forkserver swallows ``ImportError`` from
preload modules, so anything importable that can fail would silently skip the
freeze.
"""

import gc

gc.collect()  # don't make import-time cycle garbage permanent
gc.freeze()
