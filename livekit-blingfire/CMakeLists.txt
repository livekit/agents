cmake_minimum_required(VERSION 3.20)
project(LiveKitBlingFire LANGUAGES C CXX)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
if(NOT Python_VERSION VERSION_EQUAL "3.15.2")
  message(FATAL_ERROR "This source build is qualified for CPython 3.15.2")
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import sysconfig; print(sysconfig.get_config_var('Py_GIL_DISABLED') or 0)"
  OUTPUT_VARIABLE GIL_DISABLED OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
if(NOT GIL_DISABLED STREQUAL "0")
  message(FATAL_ERROR "Free-threaded Python is not qualified")
endif()
find_package(pybind11 CONFIG REQUIRED)

# The checked source projection makes the sdist build independent of Git/network.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/vendor/manifest.json" SOURCE_MANIFEST)
string(JSON EXPECTED_ARCHIVE_SHA GET "${SOURCE_MANIFEST}" archive_sha256)
string(JSON SOURCE_PREFIX GET "${SOURCE_MANIFEST}" prefix)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/vendor/blingfire.tar.gz" ARCHIVE_SHA)
if(NOT ARCHIVE_SHA STREQUAL EXPECTED_ARCHIVE_SHA)
  message(FATAL_ERROR "BlingFire source archive differs from its pinned Git projection")
endif()
file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/blingfire.tar.gz"
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/source")
set(SOURCE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/source/${SOURCE_PREFIX}")

if(APPLE)
  if((CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64") OR
      (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"))
    message(FATAL_ERROR "This source build is qualified for macOS arm64")
  endif()
  set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-mllvm,-threads=1")
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
  set(CMAKE_INSTALL_NAME_DIR "@rpath")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|x86_64)$")
    message(FATAL_ERROR "This source build is qualified for Linux x86_64/aarch64")
  endif()
  set(CMAKE_SHARED_LINKER_FLAGS "-flto=1 -Wl,--build-id=sha1")
else()
  message(FATAL_ERROR "This source build is qualified for macOS arm64 and Linux")
endif()
add_subdirectory("${SOURCE_ROOT}" upstream EXCLUDE_FROM_ALL)
pybind11_add_module(lk_blingfire src/binding.cpp)
target_link_libraries(lk_blingfire PRIVATE blingfiretokdll)
target_include_directories(lk_blingfire SYSTEM PRIVATE "${SOURCE_ROOT}/blingfiretools/blingfiretokdll")
target_compile_features(lk_blingfire PRIVATE cxx_std_17)
target_compile_options(lk_blingfire PRIVATE -Wall -Wextra -Werror)
foreach(target fsaClient blingfiretokdll lk_blingfire)
  target_compile_options(${target} PRIVATE
    "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=."
    "-fdebug-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=."
    "-ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=/build"
    "-fdebug-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=/build")
endforeach()
if(APPLE)
  set_target_properties(lk_blingfire PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path/livekit/blingfire")
else()
  set_target_properties(lk_blingfire PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN/livekit/blingfire")
endif()
install(TARGETS lk_blingfire LIBRARY DESTINATION . COMPONENT python)
install(TARGETS blingfiretokdll LIBRARY DESTINATION livekit/blingfire COMPONENT python)
install(FILES src/lk_blingfire.pyi DESTINATION . COMPONENT python)
