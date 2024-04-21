The [Chromium Embedded Framework](https://bitbucket.org/chromiumembedded/cef/) (CEF) is a simple framework for embedding Chromium-based browsers in other applications. This repository hosts a sample project called "cef-project" that can be used as the starting point for third-party applications built using CEF.

# Quick Links

* Project Page - https://bitbucket.org/chromiumembedded/cef-project
* Tutorial - https://bitbucket.org/chromiumembedded/cef/wiki/Tutorial
* Support Forum - http://www.magpcss.org/ceforum/

# Setup

First install some necessary tools and download the cef-project source code.

1\. Install [CMake](https://cmake.org/), a cross-platform open-source build system. Version 3.21 or newer is required.

2\. Install [Python](https://www.python.org/downloads/). Version 2.7.x or 3.9.x is required. If Python is not installed to the default location you can set the `PYTHON_EXECUTABLE` environment variable before running CMake (watch for errors during the CMake generation step below).

3\. Install platform-specific build tools.

* Linux: Currently supported distributions include Debian 10 (Buster), Ubuntu 18 (Bionic Beaver), and related. Ubuntu 18.04 64-bit with GCC 7.5.0+ is recommended. Newer versions will likely also work but may not have been tested. Required packages include: build-essential, libgtk-3-dev.
* MacOS: Xcode 12.2 to 15.0 building on MacOS 10.15.4 (Catalina) or newer. The Xcode command-line tools must also be installed.
* Windows: Visual Studio 2022 building on Windows 10 or newer. Windows 10/11 64-bit is recommended.

4\. Download the cef-project source code from the [Downloads page](https://bitbucket.org/chromiumembedded/cef-project/downloads) or by using [Git](https://git-scm.com/) command-line tools:

```
git clone https://bitbucket.org/chromiumembedded/cef-project.git
```

# Build

Now run CMake which will download the CEF binary distribution from the [Spotify automated builder](https://cef-builds.spotifycdn.com/index.html) and generate build files for your platform. Then build using platform build tools. For example, using the most recent tool versions on each platform:

```
cd /path/to/cef-project

# Create and enter the build directory.
mkdir build
cd build

# To perform a Linux build using a CEF binary distribution matching the host
# architecture (x64, ARM or ARM64):
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
make -j4

# To perform a MacOS build using a 64-bit CEF binary distribution:
cmake -G "Xcode" -DPROJECT_ARCH="x86_64" ..
# Then, open build\cef.xcodeproj in Xcode and select Product > Build.

# To perform a MacOS build using an ARM64 CEF binary distribution:
cmake -G "Xcode" -DPROJECT_ARCH="arm64" ..
# Then, open build\cef.xcodeproj in Xcode and select Product > Build.

# To perform a Windows build using a 32-bit CEF binary distribution:
cmake -G "Visual Studio 17" -A Win32 ..
# Then, open build\cef.sln in Visual Studio 2022 and select Build > Build Solution.

# To perform a Windows build using a 64-bit CEF binary distribution:
cmake -G "Visual Studio 17" -A x64 ..
# Then, open build\cef.sln in Visual Studio 2022 and select Build > Build Solution.

# To perform a Windows build using an ARM64 CEF binary distribution:
cmake -G "Visual Studio 17" -A arm64 ..
# Then, open build\cef.sln in Visual Studio 2022 and select Build > Build Solution.
```

CMake supports different generators on each platform. Run `cmake --help` to list all supported generators. Generators that have been tested with CEF include:

* Linux: Ninja, GCC 7.5.0+, Unix Makefiles
* MacOS: Ninja, Xcode 12.2 to 15.0
* Windows: Ninja, Visual Studio 2022+

Ninja is a cross-platform open-source tool for running fast builds using pre-installed platform toolchains (GNU, clang, Xcode or MSVC). See comments in the "third_party/cef/cef_binary_*/CMakeLists.txt" file for Ninja usage instructions.

# Examples

CEF provides a number of examples that you can use as a starting point or reference for your own CEF-based development.

* By default all example targets will be included in the project files generated using CMake.
* The CEF binary distribution includes cefsimple and cefclient sample applications.
    * The cefsimple application demonstrates the minimal implementation required for a standalone executable target and is described on the [Tutorial](https://bitbucket.org/chromiumembedded/cef/wiki/Tutorial.md) Wiki page.
    * The cefclient application demonstrates a wide range of CEF functionality most of which is documented on the [GeneralUsage](https://bitbucket.org/chromiumembedded/cef/wiki/GeneralUsage.md) Wiki page.
* The [examples directory](examples) contains example targets that demonstrate specific aspects of CEF functionality.
    * See the [examples README.md file](examples/README.md) for information about the examples targets.
    * Add `-DWITH_EXAMPLES=Off` to the cmake command-line if you do not wish to build the examples targets.

# Next Steps

Here are some activities you might want to try next to gain a better understanding of CEF:

1\. Update the CEF version used to build your local copy of cef-project:

* Visit the [Spotify automated builder](https://cef-builds.spotifycdn.com/index.html) page to see what CEF versions are available.
* Change the "CEF_VERSION" value near the top of the [top-level CMakeLists.txt file](https://bitbucket.org/chromiumembedded/cef-project/src/master/CMakeLists.txt?fileviewer=file-view-default).
* Re-run the cmake and build commands. Add `-DWITH_EXAMPLES=Off` to the cmake command-line to disable targets from the [examples directory](examples) because they may not build successfully with the new CEF version.

2\. Add your own project source code:

* Create a new "myproject" directory in the root cef-project directory (e.g. "/path/to/cef-project/myproject").
* Copy the contents of the "third_party/cef/cef_binary_*/tests/cefsimple" directory to "myproject" as a starting point.
* Add a new `add_subdirectory(myproject)` command near the end of [top-level CMakeLists.txt file](https://bitbucket.org/chromiumembedded/cef-project/src/master/CMakeLists.txt?fileviewer=file-view-default) after the existing add_subdirectory commands.
* Change the "CEF_TARGET", "CEF_HELPER_TARGET" and "CEF_HELPER_OUTPUT_NAME" values in "myproject/CMakeLists.txt" replacing "cefsimple" with "myproject".
* (Windows only) Rename the "cefclient.exe.manifest" file to "myproject.exe.manifest" in both "myproject/CMakeLists.txt" and the "myproject" directory.
* Re-run the cmake and build commands.

3\. Gain a better understanding of the cefsimple application by reading the [Tutorial](https://bitbucket.org/chromiumembedded/cef/wiki/Tutorial.md) Wiki page.

4\. Fork the cef-project repository using Bitbucket and Git to store the source code for your own CEF-based project. See the [ContributingWithGit](https://bitbucket.org/chromiumembedded/cef/wiki/ContributingWithGit.md) Wiki page for details (replace all instances of "cef" with "cef-project" in those instructions).

5\. Review the [GeneralUsage](https://bitbucket.org/chromiumembedded/cef/wiki/GeneralUsage.md) Wiki page for additional details on CEF implementation and usage.

# Support and Contributions

If you have any questions about CEF or cef-project please ask on the [CEF Forum](http://www.magpcss.org/ceforum/). If you would like to make contributions please see the "Helping Out" section of the [CEF Main Page](https://bitbucket.org/chromiumembedded/cef/).
