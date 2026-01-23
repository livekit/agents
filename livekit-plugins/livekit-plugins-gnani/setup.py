#!/usr/bin/env python
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

from setuptools import find_namespace_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Get version from version.py
about = {}
with open(os.path.join(here, "livekit", "plugins", "gnani", "version.py")) as f:
    exec(f.read(), about)

setup(
    name="livekit-plugins-gnani",
    version=about["__version__"],
    description="Agent Framework plugin for Gnani (Vachana) Speech-to-Text API.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/livekit/agents",
    author="LiveKit",
    author_email="support@livekit.io",
    license="Apache-2.0",
    packages=find_namespace_packages(include=["livekit.*"]),
    python_requires=">=3.9.0",
    install_requires=[
        "livekit-agents[codecs]>=1.3.8",
        "aiohttp>=3.8.0",
    ],
    package_data={
        "livekit.plugins.gnani": ["py.typed"],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=[
        "voice",
        "ai",
        "realtime",
        "audio",
        "livekit",
        "gnani",
        "vachana",
        "speech-to-text",
        "stt",
        "indian-languages",
    ],
)
