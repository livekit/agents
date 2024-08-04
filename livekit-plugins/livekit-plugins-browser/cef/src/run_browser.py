import sys

print("cwd: ", sys.path[0])

sys.path.insert(0, './Debug')
import lkcef_python as lkcef

print("lkcef __dict__: ", lkcef.__dict__)
print("BrowserImpl __dict__: ", lkcef.BrowserImpl.__dict__)

impl = lkcef.BrowserImpl()
impl.start()
