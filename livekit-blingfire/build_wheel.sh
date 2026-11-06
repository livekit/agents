#!/usr/bin/env bash
set -euo pipefail

# The pinned manylinux images contain 3.15.0; the extension needs the exact SDK below.
uv venv --managed-python --python 3.15.2 /opt/wheel-venv
python=/opt/wheel-venv/bin/python
uv pip install --python "$python" --require-hashes -r /project/tests/requirements.txt
uv pip install --python "$python" auditwheel==6.8.2
uv build --sdist --no-sources --python "$python" --out-dir /tmp/source-dist /project
uv build --wheel --no-sources --python "$python" --out-dir /out/raw \
  /tmp/source-dist/livekit_blingfire-1.1.0.tar.gz
case "$(uname -m)" in
  x86_64 | aarch64) platform="manylinux_2_28_$(uname -m)" ;;
  *)
    printf 'Unqualified Linux architecture\n' >&2
    exit 2
    ;;
esac
/opt/wheel-venv/bin/auditwheel repair --plat "$platform" --wheel-dir /out/repaired /out/raw/*.whl
uv pip install --python "$python" --no-deps /out/repaired/*.whl
"$python" -I /project/tests/qualify.py /out/outcomes.json
