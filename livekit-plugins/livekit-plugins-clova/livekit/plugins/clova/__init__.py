from .stt import STT
from .version import __version__

__all__ = [
    "STT",
    "__version__",
]


from livekit.agents import Plugin


class ClovaSTTPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__)

    def download_files(self):
        pass


Plugin.register_plugin(ClovaSTTPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
