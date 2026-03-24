# Copyright 2025 LiveKit, Inc.
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

import os
import pathlib

import setuptools
from setuptools import Extension

here = pathlib.Path(__file__).parent.resolve()
about = {}
with open(os.path.join(here, "livekit", "blockguard", "version.py")) as f:
    exec(f.read(), about)

setuptools.setup(
    name="livekit-blockguard",
    version=about["__version__"],
    description="Asyncio event loop blocking detector for livekit-agents",
    url="https://github.com/livekit/agents",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX",
        "Topic :: Software Development :: Debuggers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=["asyncio", "blocking", "watchdog", "livekit"],
    license="Apache-2.0",
    zip_safe=False,
    ext_modules=[
        Extension("blockguard", sources=["src/blockguard.c"]),
    ],
    package_data={"livekit.blockguard": ["py.typed"]},
    packages=setuptools.find_namespace_packages(include=["livekit.*"]),
    python_requires=">=3.10.0,<3.15",
    project_urls={
        "Documentation": "https://docs.livekit.io",
        "Website": "https://livekit.io/",
        "Source": "https://github.com/livekit/agents",
    },
)
