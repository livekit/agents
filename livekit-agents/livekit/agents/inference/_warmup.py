"""Side-effect module: imported in the forkserver preload list so the
native ``livekit-local-inference`` model singletons are loaded inside the
forkserver process. Child job processes then inherit the resident weight
pages via COW.

This must NOT be imported eagerly in the worker process itself — the worker
is not the parent of forked jobs (the forkserver is), so paging weights in
here would not benefit jobs and would cost the worker ~hundreds of MB.
"""

import livekit.local_inference as _li

_li.init_vad()
_li.init_eot()
