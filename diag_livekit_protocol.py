import importlib, traceback, os, sys
print("=== sys.path[:10] ===")
for i,p in enumerate(sys.path[:10]):
    print(i, p)

print()
# list local checkout and installed livekit folder contents
local = os.path.join(os.getcwd(), "livekit-agents", "livekit")
site_pkg = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages", "livekit")
print("Local livekit folder:", local, "exists:", os.path.isdir(local))
if os.path.isdir(local):
    print("Local livekit contents:", sorted(os.listdir(local)))

print("Site-packages livekit folder:", site_pkg, "exists:", os.path.isdir(site_pkg))
if os.path.isdir(site_pkg):
    print("Site-packages livekit contents:", sorted(os.listdir(site_pkg)))

print()
# try to import livekit.protocol and show traceback if fails
try:
    importlib.invalidate_caches()
    m = importlib.import_module("livekit.protocol")
    print("imported livekit.protocol ->", getattr(m, "__file__", getattr(m,"__path__",None)))
except Exception:
    print("FAILED to import livekit.protocol")
    traceback.print_exc()
