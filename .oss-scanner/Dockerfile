# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build image for Anthropic's OSS Scanner. Python matches the unit-test job in .github/workflows/tests.yml.
# CMAKE_POLICY_VERSION_MINIMUM lets sdists with old CMakeLists (sphn's audiopus_sys) build where no wheel exists.
FROM python:3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=3.12 \
    NLTK_DATA=/usr/share/nltk_data \
    TIKTOKEN_CACHE_DIR=/usr/share/tiktoken \
    PYTHONPATH=/src \
    CMAKE_POLICY_VERSION_MINIMUM=3.5

RUN apt-get update && apt-get install -y --no-install-recommends git-lfs \
    && rm -rf /var/lib/apt/lists/*

# CI takes the latest uv; pin one that reads this uv.lock.
RUN pip install --no-cache-dir uv==0.12.19

WORKDIR /src
COPY . /src

# A plain clone leaves LFS files as pointers. The unit tests don't read them, so a failed pull isn't fatal.
RUN (git lfs install --local --force && git lfs pull) || echo 'WARNING: git lfs pull failed; media and model files are pointers'; \
    git config --local filter.lfs.required false

RUN uv sync --locked --all-extras --dev
RUN uv run --no-sync python -c "import nltk; nltk.download('punkt_tab', download_dir='/usr/share/nltk_data')" \
    && uv run --no-sync python -c "import tiktoken; [tiktoken.get_encoding(e) for e in ('o200k_base', 'cl100k_base')]"

ENV UV_OFFLINE=1 \
    UV_NO_SYNC=1 \
    VIRTUAL_ENV=/src/.venv \
    PATH=/src/.venv/bin:$PATH

# test_room.py starts a LiveKit server with `docker run`, so it can't run here.
RUN --network=none timeout 20m uv run --no-sync pytest --unit --audio_eot --ignore=tests/test_room.py \
    || echo 'WARNING: baseline tests failed'

CMD ["bash"]
