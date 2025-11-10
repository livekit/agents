import sys, importlib, os, pkgutil
print("=== sys.path (first 10 entries) ===")
for i,p in enumerate(sys.path[:10]):
    print(i, p)

print()
# List any locations on sys.path that contain a 'livekit' folder
print("=== Locations that contain a 'livekit' folder (if any) ===")
for p in sys.path:
    if os.path.isdir(os.path.join(p, "livekit")):
        print("FOUND livekit directory at:", os.path.join(p, "livekit"))

print()
# Try to import the local 'livekit' package and display info
try:
    m = importlib.import_module("livekit")
    print("import succeeded. livekit __file__/__path__:", getattr(m, "__file__", getattr(m, "__path__", None)))
    print("has rtc?", hasattr(m, "rtc"))
except Exception as e:
    print("import error:", repr(e))

print()
# List files under the local livekit folder (if present)
local = os.path.join(os.getcwd(), "livekit-agents", "livekit")
print("Checking local folder:", local)
if os.path.isdir(local):
    print("Local livekit folder contents:")
    for name in sorted(os.listdir(local)):
        print("  -", name)
else:
    print("Local livekit folder not found at expected path.")
