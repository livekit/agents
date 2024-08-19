# flake8: noqa

import sys

print("cwd: ", sys.path[0])

sys.path.insert(0, "./Debug")
import lkcef_python as lkcef

print("lkcef __dict__: ", lkcef.__dict__)
print("BrowserImpl __dict__: ", lkcef.BrowserImpl.__dict__)


def _context_initialized():
    opts = lkcef.BrowserOptions()
    opts.framerate = 30

    def _browser_created(browser_impl):
        print("run_browser.py - Browser created")

    opts.created_callback = _browser_created

    def on_paint(frame_data):
        pass

    opts.paint_callback = on_paint

    app.create_browser("http://www.livekit.io", opts)
    print("run_browser.py - Context initialized")


opts = lkcef.AppOptions()
opts.dev_mode = True
opts.initialized_callback = _context_initialized

app = lkcef.BrowserApp(opts)
app.run()
