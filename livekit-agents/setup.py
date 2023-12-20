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
import setuptools.command.build_py


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
    url="https://github.com/livekit/python-agents",
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
        "click>=8.1.7",
        "livekit>=0.6.0",
        "livekit-api>=0.1.3",
        "livekit-protocol>=0.1.0",
        "livekit-protocol",
        "websockets>=12.0",
    ],
    package_data={},
    project_urls={
        "Documentation": "https://docs.livekit.io",
        "Website": "https://livekit.io/",
        "Source": "https://github.com/livekit/python-agents",
    },
)
