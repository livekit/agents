cmake_minimum_required(VERSION 3.19)
set(CMAKE_CONFIGURATION_TYPES Debug Release)

project(livekit-aec)

set(CMAKE_CXX_STANDARD 17)

# who use this c++ extension on vscode instead of clangd?
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
    abseil
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG        20250127.0
)

FetchContent_MakeAvailable(abseil)

file(GLOB_RECURSE SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_definitions(-DWEBRTC_APM_DEBUG_DUMP=0)

if(APPLE)
    add_definitions(-DWEBRTC_MAC -DWEBRTC_POSIX -DRTC_DISABLE_LOGGING)
elseif(UNIX AND NOT APPLE)
    add_definitions(-DWEBRTC_LINUX -DWEBRTC_POSIX -DRTC_DISABLE_LOGGING)
elseif(WIN32)
    add_definitions(-DWEBRTC_WIN -DRTC_DISABLE_LOGGING)
endif()

add_library(livekit-aec STATIC ${SRC})

target_include_directories(livekit-aec PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(livekit-aec PRIVATE absl::base)
target_include_directories(livekit-aec BEFORE PUBLIC $<BUILD_INTERFACE:${absl_SOURCE_DIR}>)

# Python bindings
find_package(PythonInterp REQUIRED)
find_package(pybind11 REQUIRED)

message(STATUS "Using Python: ${PYTHON_EXECUTABLE}")

pybind11_add_module(lkaec_python ${CMAKE_CURRENT_SOURCE_DIR}/aec_python.cc)
target_link_libraries(lkaec_python PUBLIC livekit-aec)
