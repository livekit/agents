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
import re
import subprocess
import sys
from pathlib import Path

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext

here = pathlib.Path(__file__).parent.resolve()
about = {}
with open(os.path.join(here, "livekit", "plugins", "browser", "version.py"), "r") as f:
    exec(f.read(), about)


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        print(f"cmake_args: {cmake_args}")

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        self.build_temp = Path(self.build_temp) / ext.name
        if not self.build_temp.exists():
            self.build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=self.build_temp, check=True
        )
        subprocess.run(["cmake", "--build", "."], cwd=self.build_temp, check=True)

        build_output = self.build_temp / "src" / cfg

        for f in build_output.iterdir():
            if f.suffix == ".so":
                self.copy_file(f, extdir / f.name)

        if sys.platform.startswith("darwin"):
            # on macos, copy the dummy app
            app = build_output / "lkcef_app.app"
            self.copy_tree(
                app,
                str(
                    extdir
                    / "livekit"
                    / "plugins"
                    / "browser"
                    / "resources"
                    / "lkcef_app.app"
                ),
            )


setuptools.setup(
    name="livekit-plugins-browser",
    version=about["__version__"],
    description="Chromium Embedded Framework (CEF) for LiveKit Agents",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/livekit/agents",
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
    keywords=["webrtc", "realtime", "audio", "video", "livekit"],
    license="Apache-2.0",
    ext_modules=[CMakeExtension("lkcef_python")],
    cmdclass={"build_ext": CMakeBuild},
    packages=setuptools.find_namespace_packages(include=["livekit.*"]),
    python_requires=">=3.9.0",
    install_requires=["livekit-agents>=0.8.0"],
    package_data={
        "livekit.plugins.browser": ["py.typed"],
        "livekit.plugins.browser.resources": ["**", "lkcef_app.app"],
    },
    project_urls={
        "Documentation": "https://docs.livekit.io",
        "Website": "https://livekit.io/",
        "Source": "https://github.com/livekit/agents",
    },
)
