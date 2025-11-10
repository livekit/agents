import importlib, traceback, os
print("=== Trying to import livekit.rtc and livekit.api ===")
for name in ("livekit.rtc", "livekit.api"):
    try:
        importlib.invalidate_caches()
        m = importlib.import_module(name)
        print(f"imported {name} ->", getattr(m, "__file__", getattr(m, "__path__", None)))
    except Exception:
        print(f"FAILED to import {name}")
        traceback.print_exc()

print()
local = os.path.join(os.getcwd(), "livekit-agents", "livekit")
print("Local livekit folder:", local)
if os.path.isdir(local):
    print("Local livekit contents:")
    for n in sorted(os.listdir(local)):
        print("  -", n)
else:
    print("Local livekit folder not found")

print()
site_pkg = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages", "livekit")
print("Site-packages livekit folder:", site_pkg)
if os.path.isdir(site_pkg):
    print("Site-packages livekit contents:")
    for n in sorted(os.listdir(site_pkg)):
        print("  -", n)
else:
    print("Site-packages livekit folder not found")
