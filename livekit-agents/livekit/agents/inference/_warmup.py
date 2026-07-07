"""Side-effect module: imported in the forkserver preload list so the
native ``livekit-local-inference`` model singletons are loaded inside the
forkserver process. Child job processes then inherit the resident weight
pages via COW.
"""

import livekit.local_inference as _li

_li.init_vad()
_li.init_eot()
