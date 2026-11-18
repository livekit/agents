"""The page, on http://localhost:8000: a static file server for web/."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __name__ == "__main__":
    handler = partial(SimpleHTTPRequestHandler, directory=Path(__file__).parent / "web")
    ThreadingHTTPServer(("localhost", 8000), handler).serve_forever()
