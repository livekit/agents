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

    def _on_paint(frame_data):
        pass

    opts.paint_callback = _on_paint

    def _on_closed():
        print("run_browser.py - Browser closed")

    opts.close_callback = _on_closed

    app.create_browser("http://www.livekit.io", opts)
    print("run_browser.py - Context initialized")


opts = lkcef.AppOptions()
opts.dev_mode = True
opts.initialized_callback = _context_initialized
opts.framework_path = "/Users/theomonnom/livekit/agents/livekit-plugins/livekit-plugins-browser/cef/src/Debug/lkcef_app.app/Contents/Frameworks/Chromium Embedded Framework.framework"
opts.main_bundle_path = "/Users/theomonnom/livekit/agents/livekit-plugins/livekit-plugins-browser/cef/src/Debug/lkcef_app.app"
opts.subprocess_path = "/Users/theomonnom/livekit/agents/livekit-plugins/livekit-plugins-browser/cef/src/Debug/lkcef_app.app/Contents/Frameworks/lkcef Helper.app/Contents/MacOS/lkcef Helper"

app = lkcef.BrowserApp(opts)
app.run()
