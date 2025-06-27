cmake_minimum_required(VERSION 3.19)
set(CMAKE_CONFIGURATION_TYPES Debug Release)

project(livekit_blingfire)
set_property(GLOBAL PROPERTY OS_FOLDERS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PythonInterp REQUIRED)
find_package(pybind11 REQUIRED)

message(STATUS "Using Python: ${PYTHON_EXECUTABLE}")

include(FetchContent)
FetchContent_Declare(blingfire
                     GIT_REPOSITORY https://github.com/microsoft/BlingFire)
FetchContent_MakeAvailable(blingfire)

include_directories(
  ${blingfire_SOURCE_DIR}
)

pybind11_add_module(livekit_blingfire main.cpp)
target_link_libraries(livekit_blingfire PRIVATE blingfiretokdll_static
                                                fsaClient)
