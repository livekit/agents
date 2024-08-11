import _winapi
import tempfile
import os


_mmap_counter = itertools.count()

BUFSIZE = 8192



def arbitrary_address():
    return tempfile.mktemp(prefix=r'\\.\pipe\pyc-%d-%d-' %
                               (os.getpid(), next(_mmap_counter)), dir="")

def AsyncDuplex:
    def __init__(self):
        address = arbitrary_address()
        openmode = _winapi.PIPE_ACCESS_DUPLEX
        access = _winapi.GENERIC_READ | _winapi.GENERIC_WRITE

        h1 = _winapi.CreateNamedPipe(
            address, openmode | _winapi.FILE_FLAG_OVERLAPPED |
            _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE,
            _winapi.PIPE_TYPE_MESSAGE | _winapi.PIPE_READMODE_MESSAGE |
            _winapi.PIPE_WAIT,
            1, BUFSIZE, BUFSIZE, _winapi.NMPWAIT_WAIT_FOREVER,
            # default security descriptor: the handle cannot be inherited
            _winapi.NULL
            )
        h2 = _winapi.CreateFile(
            address, access, 0, _winapi.NULL, _winapi.OPEN_EXISTING,
            _winapi.FILE_FLAG_OVERLAPPED, _winapi.NULL
            )
        _winapi.SetNamedPipeHandleState(
            h2, _winapi.PIPE_READMODE_MESSAGE, None, None
            )

        overlapped = _winapi.ConnectNamedPipe(h1, overlapped=True)
        _, err = overlapped.GetOverlappedResult(True)
        #assert err == 0

        #c1 = PipeConnection(h1, writable=duplex)
        #c2 = PipeConnection(h2, readable=duplex)
        pass