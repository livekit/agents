import os
import pathlib

import setuptools
import setuptools.command.build_py

here = pathlib.Path(__file__).parent.resolve()
about = {}
with open(os.path.join(here, "livekit", "plugins", "playht", "version.py"), "r") as f:
    exec(f.read(), about)


setuptools.setup(
    name="livekit-plugins-playht",
    version=about["__version__"],
    description="Agent Framework plugin for voice synthesis with PlayHT's API.",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/livekit/agents",
    cmdclass={},
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=["webrtc", "realtime", "audio", "livekit", "playHT"],
    license="Apache-2.0",
    packages=setuptools.find_namespace_packages(include=["livekit.*"]),
    python_requires=">=3.9.0",
    install_requires=[
        "livekit-agents[codecs]>=0.8.0",
        "pyht",
        "aiohttp",
        "livekit",
    ],
    package_data={"livekit.plugins.playht": ["py.typed"]},
    project_urls={
        "Documentation": "https://docs.livekit.io",
        "Website": "https://livekit.io/",
        "Source": "https://github.com/livekit/agents",
    },
)
