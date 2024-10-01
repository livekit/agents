# Copyright 2023 LiveKit, Inc.
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

here = pathlib.Path(__file__).parent.resolve()
about = {}
with open(os.path.join(here, "livekit", "agents", "version.py"), "r") as f:
    exec(f.read(), about)


setuptools.setup(
    name="livekit-agents",
    version=about["__version__"],
    description="LiveKit Python Agents",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/livekit/agents",
    cmdclass={},
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=["webrtc", "realtime", "audio", "video", "livekit", "agents", "AI"],
    license="Apache-2.0",
    packages=setuptools.find_namespace_packages(include=["livekit.*"]),
    python_requires=">=3.9.0",
    install_requires=[
        "click~=8.1",
        "livekit>=0.17.2",
        "livekit-api~=0.6",
        "livekit-protocol~=0.6",
        "protobuf>=3",
        "pyjwt>=2.0.0",
        "types-protobuf>=4,<5",
        "watchfiles~=0.22",
        "psutil~=5.9",
        "aiohttp~=3.10",
        "typing-extensions~=4.12",
    ],
    extras_require={
        ':sys_platform=="win32"': [
            "colorama"
        ],  # fix logs color on windows (devmode only)
        ':sys_platform!="win32"': [
            "aiodns~=3.2"
        ],  # use default aiohttp resolver on windows
        "codecs": ["av>=11.0.0"],
        "images": ["pillow~=10.3.0"],
    },
    package_data={"livekit.agents": ["py.typed"]},
    project_urls={
        "Documentation": "https://docs.livekit.io",
        "Website": "https://livekit.io/",
        "Source": "https://github.com/livekit/agents",
    },
)
