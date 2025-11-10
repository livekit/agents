import pathlib
import re

root = pathlib.Path(".").resolve()
py_files = list(root.rglob("*.py"))

pattern_replacements = [
    (re.compile(r'from\s+\.types\b'), 'from .agents_types'),
    (re.compile(r'from\s+\.\.types\b'), 'from ..agents_types'),
    (re.compile(r'from\s+livekit\.agents\.types\b'), 'from livekit.agents.agents_types'),
]

changed = []
for p in py_files:
    text = p.read_text(encoding="utf-8")
    new_text = text
    for pat, repl in pattern_replacements:
        new_text = pat.sub(repl, new_text)
    if new_text != text:
 
        bak = p.with_suffix(p.suffix + ".bak")
        bak.write_text(text, encoding="utf-8")
        p.write_text(new_text, encoding="utf-8")
        changed.append(str(p.relative_to(root)))

print("Files updated (and .bak created):")
for f in changed:
    print(" -", f)
if not changed:
    print("No files needed updating.")
